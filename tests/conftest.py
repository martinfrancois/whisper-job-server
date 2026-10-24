"""Test setup: import app from a scratch working directory and replace the ML stack with fakes.

app.py creates jobs/ and logs/ in the working directory at import, so the session
first changes into a temporary directory. torch, torchaudio and whisper are never
imported for real: fake modules go into sys.modules before the worker loads them,
so no model is downloaded and the tests run without the ML packages installed.
"""

import importlib
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_WAV = REPO_ROOT / "examples" / "tone.wav"
FAKE_TRANSCRIPT = " hello from the fake model"


@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("server")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(workdir)
        patch.syspath_prepend(str(REPO_ROOT))
        for name in ("WHISPER_JOB_SERVER_TOKEN", "WHISPER_JOB_SERVER_CORS_ORIGINS", "WHISPER_JOB_SERVER_MAX_UPLOAD_MB"):
            patch.delenv(name, raising=False)
        yield importlib.import_module("app")


@pytest.fixture
def app(app_module, monkeypatch):
    """A fresh in-memory job table and an empty queue for every test."""
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "API_TOKEN", None)
    shutil.rmtree(app_module.JOB_BASE, ignore_errors=True)
    os.makedirs(app_module.JOB_BASE)
    while not app_module.job_queue.empty():
        app_module.job_queue.get_nowait()
    return app_module


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    # No context manager: the lifespan hook expects process_startup() to have run.
    return TestClient(app.app)


class FakeWhisperModel:
    def transcribe(self, audio_path, language="en", fp16=False):
        assert os.path.exists(audio_path)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": FAKE_TRANSCRIPT}]}


def _fake_torch_modules():
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.device = lambda name: name
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace()
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    nn.functional = functional
    torch.nn = nn
    torchaudio = types.ModuleType("torchaudio")
    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda name: FakeWhisperModel()
    return {
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "torchaudio": torchaudio,
        "whisper": whisper,
    }


@pytest.fixture
def fake_ml_stack(app, monkeypatch):
    """Install fake torch, torchaudio and whisper modules for one test."""
    for name, module in _fake_torch_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    for attr in ("torch", "torchaudio", "F", "whisper"):
        monkeypatch.setattr(app, attr, None)
    return app
