# Queue Whisperer

A self-hosted web server that queues audio and video files and transcribes them with OpenAI's Whisper.

## Table of contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Installation](#installation)
  - [FFmpeg](#ffmpeg)
  - [Virtual environment](#virtual-environment)
  - [Python dependencies](#python-dependencies)
  - [PyTorch with CUDA support](#pytorch-with-cuda-support)
  - [Diarization assets from Hugging Face](#diarization-assets-from-hugging-face)
- [Usage](#usage)
- [Configuration](#configuration)
- [Security](#security)
- [Example](#example)
- [Job management](#job-management)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [License](#license)

## What it does

`app.py` is a FastAPI application with a web page at `/`. You upload a file,
pick a mode and a language, and the server converts the file to 16 kHz mono
WAV with FFmpeg, runs Whisper `large-v2` on it and stores two transcripts, one
plain and one with timestamps. Jobs run one after another. The page shows
progress over a WebSocket and offers both transcripts for download.

Three modes:

- Single speaker: one Whisper pass over the whole file.
- Stereo: the two channels are transcribed separately as Speaker 1 and
  Speaker 2 and merged by timestamp. Use it for calls recorded with one
  speaker per channel.
- Speaker detection: `pyannote` diarization splits the audio by speaker, each
  segment is transcribed, and speaker labels are cleaned up with speaker
  embeddings from SpeechBrain.

## Requirements

- Python 3.10 or newer. The pinned packages need at least 3.10; the tests run
  on 3.12 and 3.13.
- FFmpeg and FFprobe on the `PATH`, or their paths in `FFMPEG_BIN` and
  `FFPROBE_BIN`.
- About 3 GB of disk for the Whisper `large-v2` model, which downloads on the
  first job, and enough RAM to hold it. A CUDA GPU is optional; the example
  below was run with the CPU build of PyTorch.
- A Hugging Face account for speaker detection mode only.

## Installation

### FFmpeg

- Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html) or
  `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Debian or Ubuntu: `sudo apt update && sudo apt install ffmpeg`
- Fedora: `sudo dnf install ffmpeg`

### Virtual environment

```bash
python3 -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

On macOS with Homebrew, `brew install python@3.13` gives you a recent
interpreter; create the virtual environment with
`"$(brew --prefix python@3.13)/bin/python3.13" -m venv venv`.

### Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every package. On a machine without a GPU, install
the CPU build of PyTorch first so that pip keeps it, then the rest:

```bash
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### PyTorch with CUDA support

If you have a CUDA-capable GPU, pick the matching build on the
[PyTorch Get Started](https://pytorch.org/get-started/locally/) page and run
the command it shows, for example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Keep the CUDA version at or below the CUDA toolkit installed on the machine.

### Diarization assets from Hugging Face

Speaker detection mode needs gated models from Hugging Face.

1. Create an account on [Hugging Face](https://huggingface.co/).
2. Accept the licenses of `pyannote/speaker-diarization-community-1` and
   `pyannote/segmentation`.
3. Generate a fine-grained access token with read permission on
   `pyannote/speaker-diarization-community-1`, `pyannote/segmentation` and
   `speechbrain/spkrec-ecapa-voxceleb`.
4. Set the token on the machine that will clone the models:

   ```bash
   # On Windows:
   set HUGGING_FACE_HUB_TOKEN=your_token_here

   # On macOS/Linux:
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

5. Install [Git LFS](https://git-lfs.com/) and clone the pipeline next to
   `app.py`. If git asks for a password, use the access token.

   ```bash
   git lfs install
   git clone https://hf.co/pyannote/speaker-diarization-community-1 pyannote-speaker-diarization-community-1
   ```

6. If you prepared the clone on another machine, copy the
   `pyannote-speaker-diarization-community-1` folder next to `app.py` before
   starting the server.

The server loads the pipeline from that folder when it exists. Otherwise it
downloads the pipeline from Hugging Face with `HUGGING_FACE_HUB_TOKEN`.

## Usage

```bash
python app.py
```

The server listens on `http://127.0.0.1:8000`. Open that address, upload an
audio or video file, choose the mode and the language, and watch the job in
the list below the form. When it is done, download the transcript with or
without timestamps.

The same actions are available over HTTP:

| Method and path | Purpose |
|---|---|
| `POST /upload` | Multipart form with `file`, `mode` (`one`, `stereo`, `diarization`) and `language` (a Whisper language code, default `en`). Returns the job id. |
| `GET /jobs` | All jobs with status, progress and ETA as JSON. |
| `GET /download/{job_id}/without` | Plain transcript of a finished job. |
| `GET /download/{job_id}/with` | Transcript with `[start - end]` timestamps. |
| `GET /restart/{job_id}` | Queue a failed or cancelled job again. |
| `GET /remove/{job_id}` | Delete a job and its files. |
| `WS /ws` | Sends the job table as JSON once per second. |

Uploads must be a plain filename with one of these extensions: 3gp, aac, aif,
aiff, amr, avi, flac, m4a, m4v, mkv, mov, mp3, mp4, mpeg, mpg, oga, ogg, opus,
wav, webm, wma, wmv. The server stores the file under a generated name inside
`jobs/<job id>/` and keeps the original name for display and for the download
filename.

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_JOB_SERVER_HOST` | `127.0.0.1` | Address to listen on. `0.0.0.0` exposes the server to the network; read [Security](#security) first. |
| `WHISPER_JOB_SERVER_PORT` | `8000` | Port to listen on. |
| `WHISPER_JOB_SERVER_TOKEN` | unset | When set, `POST /upload`, `GET /restart/...` and `GET /remove/...` require the header `Authorization: Bearer <token>`. |
| `WHISPER_JOB_SERVER_MAX_UPLOAD_MB` | `2048` | Uploads larger than this are rejected with status 413. |
| `WHISPER_JOB_SERVER_CORS_ORIGINS` | unset | Comma-separated browser origins allowed to call the API from another site. Unset means no CORS headers, which is all the built-in page needs. |
| `HUGGING_FACE_HUB_TOKEN` | unset | Token for the gated diarization models. |
| `FFMPEG_BIN`, `FFPROBE_BIN` | `ffmpeg`, `ffprobe` | Paths to the FFmpeg binaries. |

## Security

The server has no user management. Anyone who can reach the port can upload
files, list jobs and download transcripts. By default it therefore listens on
the loopback interface only, so only programs on the same machine can reach
it.

If you set `WHISPER_JOB_SERVER_TOKEN`, every request that creates, restarts
or removes a job must carry `Authorization: Bearer <token>`. The web page asks
for the token the first time the server answers 401 and keeps it for the
browser tab. The token does not cover `GET /jobs`, the downloads or the
WebSocket, so it stops strangers from submitting work, not from reading
results.

To expose the server beyond the machine, put it behind a reverse proxy that
authenticates users and terminates TLS, and point the proxy at the loopback
address. Do not set `WHISPER_JOB_SERVER_HOST=0.0.0.0` on a network you do not
control.

Uploaded files are stored under a server-generated name, never under the name
the client sent, and the server checks that the resulting path stays inside
the `jobs/` folder. Uploads are capped by `WHISPER_JOB_SERVER_MAX_UPLOAD_MB`.

## Example

`examples/tone.wav` is a one-second 440 Hz tone, 16 kHz mono, written by
`examples/make_wav.py` with the standard library `wave` module. It contains no
speech, so do not expect a real transcript; on a CPU-only run Whisper wrote
`Beep.` for it. It exercises the whole path: upload, FFmpeg conversion, model
load, transcription, download.

Start the server in one terminal:

```bash
python app.py
```

In another terminal:

```bash
curl -F "file=@examples/tone.wav" -F mode=one -F language=en http://127.0.0.1:8000/upload
# {"job_id":"<id>","message":"File uploaded and job queued."}

curl http://127.0.0.1:8000/jobs
# {"<id>":{"status":"pending", ... "original_filename":"tone.wav", ...}}
```

The first job downloads the 2.9 GB `large-v2` model. When `GET /jobs` reports
`"status":"done"`, fetch the transcripts:

```bash
curl -OJ http://127.0.0.1:8000/download/<id>/without   # tone_transcription.txt
curl -OJ http://127.0.0.1:8000/download/<id>/with      # tone_transcription_timestamps.txt
```

With a token set, add `-H "Authorization: Bearer <token>"` to the upload
call.

To regenerate the example file:

```bash
python3 examples/make_wav.py
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use FastAPI's `TestClient` and replace `torch`, `torchaudio` and
`whisper` with small fakes, so they need neither the ML packages nor a model
download. They cover the filename and extension checks, the upload limit, the
token, the empty job list, and one full worker run on `examples/tone.wav`
through FFmpeg with a fake model. That last test needs `ffmpeg` on the `PATH`
and is skipped otherwise. CI runs the same command on Ubuntu with Python 3.12.

## Job management

- Job status and progress are pushed over the WebSocket once per second.
- Jobs are processed one at a time in the order they were submitted.
- A job that was running when the server stopped is marked failed at the next
  start and can be restarted.
- `GET /remove/{job_id}` deletes the job folder, including the upload and the
  transcripts.
- The server keeps a running average of transcription time per mode and
  language in `runtime_history.json` and uses it for the ETA.

## Troubleshooting

- FFmpeg errors: check that `ffmpeg` and `ffprobe` run from the shell that
  starts the server, or set `FFMPEG_BIN` and `FFPROBE_BIN`.
- First job takes long: the `large-v2` model, about 2.9 GB, downloads into
  Whisper's cache on the first run.
- CUDA errors: check `nvidia-smi` and that the installed PyTorch build matches
  the driver's CUDA version.
- Speaker detection fails to load the pipeline: check that
  `HUGGING_FACE_HUB_TOKEN` is set and that the licenses in the installation
  section are accepted.
- Details of a failed job are in the `logs` folder next to `app.py`.

## Limitations

- The model name is fixed to `large-v2` and is loaded again for every job; a
  stereo job loads it three times.
- The web page's WebSocket uses `ws://`, so live updates do not work when the
  page is served over HTTPS through a proxy. Polling `GET /jobs` still works.
- The web page offers English and German; the API accepts any Whisper language
  code.
- The bearer token protects the endpoints that change jobs, not the ones that
  read them.
- Speaker detection needs the gated Hugging Face models, and the server's own
  default estimate puts it at three times the runtime of single speaker mode.

## License

MIT, see `LICENSE`.
