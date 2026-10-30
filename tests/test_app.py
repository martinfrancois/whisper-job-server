import errno
import io
import math
import os
import shutil
import struct
import sys
import threading
import wave

import pytest

from conftest import EXAMPLE_WAV, FAKE_TRANSCRIPT


def upload(client, filename="tone.wav", data=None, headers=None, mode="one"):
    if data is None:
        data = EXAMPLE_WAV.read_bytes()
    return client.post(
        "/upload",
        files={"file": (filename, data, "application/octet-stream")},
        data={"mode": mode, "language": "en"},
        headers=headers or {},
    )


def test_job_listing_on_empty_queue(client):
    response = client.get("/jobs")
    assert response.status_code == 200
    assert response.json() == {}


def test_upload_stores_file_under_generated_name(app, client):
    response = upload(client, filename="my recording.wav")
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    job = app.jobs[job_id]
    assert job["status"] == "pending"
    assert job["original_filename"] == "my recording.wav"
    stored = job["input_file"]
    assert os.path.dirname(stored) == os.path.join(app.JOB_BASE, job_id)
    assert os.path.basename(stored) != "my recording.wav"
    assert stored.endswith(".wav")
    assert os.path.getsize(stored) == EXAMPLE_WAV.stat().st_size
    assert app.job_queue.get_nowait()["job_id"] == job_id
    assert client.get("/jobs").json()[job_id]["mode"] == "one"


@pytest.mark.parametrize("filename", ["../../etc/passwd.wav", "..\\..\\evil.wav", "sub/dir.wav", "..", ""])
def test_traversal_filename_rejected(app, client, filename):
    response = upload(client, filename=filename)
    assert response.status_code in (400, 422), response.text
    assert app.jobs == {}
    assert os.listdir(app.JOB_BASE) == []


@pytest.mark.parametrize("filename", ["notes.txt", "script.py", "archive.zip", "noextension"])
def test_extension_outside_allowlist_rejected(app, client, filename):
    response = upload(client, filename=filename)
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]
    assert app.jobs == {}
    assert os.listdir(app.JOB_BASE) == []


def test_oversize_upload_rejected(app, client, monkeypatch):
    monkeypatch.setattr(app, "MAX_UPLOAD_MB", 1)
    response = upload(client, data=b"\0" * (app.BYTES_PER_MB + 1))
    assert response.status_code == 413
    assert "1 MB" in response.json()["detail"]
    assert app.jobs == {}
    assert os.listdir(app.JOB_BASE) == []
    assert app.job_queue.empty()


def test_upload_at_limit_accepted(app, client, monkeypatch):
    monkeypatch.setattr(app, "MAX_UPLOAD_MB", 1)
    response = upload(client, data=b"\0" * app.BYTES_PER_MB)
    assert response.status_code == 200, response.text


def test_failed_write_leaves_no_job_folder(app, client, monkeypatch):
    class FullDisk:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, chunk):
            raise OSError(errno.ENOSPC, "No space left on device")

    # app.py looks up open() in its module globals before the builtins.
    monkeypatch.setattr(app, "open", lambda *args, **kwargs: FullDisk(), raising=False)
    with pytest.raises(OSError):
        upload(client)
    assert app.jobs == {}
    assert os.listdir(app.JOB_BASE) == []
    assert app.job_queue.empty()


def test_token_required_when_configured(app, client, monkeypatch):
    monkeypatch.setattr(app, "API_TOKEN", "s3cret")
    assert upload(client).status_code == 401
    assert upload(client, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/restart/unknown").status_code == 401
    assert client.get("/remove/unknown").status_code == 401
    assert client.get("/jobs").status_code == 200
    ok = upload(client, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200, ok.text
    # Auth runs before the job lookup, so an unknown job answers 404 once the token is right.
    assert client.get("/restart/unknown", headers={"Authorization": "Bearer s3cret"}).status_code == 404


def test_token_not_required_when_unset(app, client):
    assert app.API_TOKEN is None
    assert upload(client).status_code == 200
    assert client.get("/restart/unknown").status_code == 404
    assert client.get("/remove/unknown").status_code == 404


def test_cors_headers_absent_by_default(client):
    response = client.get("/jobs", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_worker_transcribes_example_wav_with_fake_model(fake_ml_stack, client):
    app = fake_ml_stack
    job_id = upload(client, filename="tone.wav").json()["job_id"]
    queued = app.job_queue.get_nowait()
    app.process_job(app.jobs, job_id, queued["file_location"], queued["mode"], queued["language"], threading.Event())

    job = app.jobs[job_id]
    assert job["status"] == "done", job.get("error")
    assert app.torch is not None and app.whisper is not None
    job_dir = os.path.join(app.JOB_BASE, job_id)
    assert os.path.exists(os.path.join(job_dir, "input.wav"))
    with open(job["result_files"]["without"], encoding="utf-8") as f:
        assert f.read() == FAKE_TRANSCRIPT
    with open(job["result_files"]["with"], encoding="utf-8") as f:
        assert f.read() == f"[0.00s - 1.00s] {FAKE_TRANSCRIPT}"

    download = client.get(f"/download/{job_id}/without")
    assert download.status_code == 200
    assert download.text == FAKE_TRANSCRIPT
    assert 'filename="tone_transcription.txt"' in download.headers["content-disposition"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_stereo_worker_preserves_both_channel_signals(fake_ml_stack, client, monkeypatch):
    app = fake_ml_stack
    channels = [
        [int(12000 * math.sin(2 * math.pi * frequency * i / 16000)) for i in range(16000)]
        for frequency in (440, 880)
    ]
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<hh", left, right) for left, right in zip(*channels)))

    transcribed = []
    loaded_models = []

    class ChannelCheckingModel:
        def transcribe(self, audio_path, language="en", fp16=False):
            assert self is loaded_models[0]
            assert language == "en"
            assert fp16 is False
            channel = len(transcribed)
            with wave.open(audio_path, "rb") as wav:
                assert wav.getnchannels() == 1
                assert wav.getframerate() == 16000
                assert wav.getsampwidth() == 2
                samples = struct.unpack("<16000h", wav.readframes(wav.getnframes()))
            assert list(samples) == channels[channel]
            transcribed.append(channel)
            return {"segments": [{"start": 0.0, "end": 1.0, "text": f"channel {channel}"}]}

    def load_model(name):
        assert name == "large-v2"
        model = ChannelCheckingModel()
        loaded_models.append(model)
        return model

    monkeypatch.setattr(sys.modules["whisper"], "load_model", load_model)
    response = upload(client, filename="stereo.wav", data=audio.getvalue(), mode="stereo")
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    queued = app.job_queue.get_nowait()
    app.process_job(app.jobs, job_id, queued["file_location"], queued["mode"], queued["language"], threading.Event())

    job = app.jobs[job_id]
    assert len(loaded_models) == 1
    assert job["status"] == "done", job.get("error")
    assert transcribed == [0, 1]
    download = client.get(f"/download/{job_id}/without")
    assert download.status_code == 200
    assert download.text == "Speaker 1: channel 0\nSpeaker 2: channel 1"
