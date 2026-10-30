# Queue Whisperer

Queue Whisperer is a self-hosted web application that manages audio transcription jobs using OpenAI's Whisper.
It features a built-in job queue system that supports multiple transcription modes, allowing you to submit, process, and monitor transcription tasks in real time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [FFmpeg](#ffmpeg)
  - [Virtual Environment Setup](#virtual-environment-setup)
  - [Python Dependencies](#python-dependencies)
  - [PyTorch with CUDA Support](#pytorch-with-cuda-support)
  - [Diarization Assets (Hugging Face)](#diarization-assets-hugging-face)
- [Usage](#usage)
- [HTTP API](#http-api)
- [Configuration](#configuration)
- [Security](#security)
- [Example](#example)
- [Tests](#tests)
- [Job Management](#job-management)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
- [License](#license)

## Overview

Queue Whisperer is built with FastAPI and leverages OpenAI's Whisper model to transcribe audio files.
It supports:

- **Single Speaker Mode:** For audio with one speaker.
- **Stereo Mode:** For dual-channel audio.
- **Speaker Detection:** AI-based detection and transcription of multiple speakers.

At its core, Queue Whisperer is designed to manage a job queue for transcription tasks, enabling you to submit jobs, track their progress, and retrieve results once processing is complete.

## Features

- **Multiple Transcription Modes:**
  - **Single Speaker Mode:** For audio with one speaker.
  - **Stereo Mode:** For dual-channel audio.
  - **Speaker Detection:** Detects and transcribes multiple speakers using AI.
- **Job Queue System:** Submit transcription jobs and monitor their status in real time.
- **Download Options:** Retrieve transcriptions with or without timestamps.
- **Developer-Friendly:** Designed for self-hosting and easy customization.

## Prerequisites

- **Python:** Version 3.10.0 up to (but not including) 3.14. Any 3.13.x release is supported, but Python 3.14+ is not. CI runs the tests on Python 3.12 only.
- **FFmpeg:** Installed and available in your system PATH (both `ffmpeg` and `ffprobe`).
- **GPU (Optional):** A CUDA-capable GPU is recommended for improved performance.
- **Hugging Face Account:** Required for using speaker detection mode.

## Installation

### FFmpeg

Ensure FFmpeg is installed and accessible in your system PATH.

- **Windows:**  
  Download from the [FFmpeg website](https://ffmpeg.org/download.html) or install via Chocolatey:
  ```bash
  choco install ffmpeg
  ```
- **macOS:**
  ```bash
  brew install ffmpeg
  ```
- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```
- **Linux (Fedora/CentOS):**
  ```bash
  sudo dnf install ffmpeg
  ```

### Virtual Environment Setup

Create and activate a virtual environment using a supported Python interpreter (3.10 ≤ version < 3.14):

```bash
python3.13 -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### macOS (Homebrew)

To install Python 3.13 alongside the system Python and create an isolated environment for this project:

```bash
brew install python@3.13
PY313="$(brew --prefix python@3.13)/bin/python3.13"
"$PY313" -m venv venv
source venv/bin/activate
python -m pip install -U pip wheel setuptools
```

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### PyTorch with CUDA Support

If you have a CUDA-capable GPU, install PyTorch with CUDA support. Visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page and choose the options appropriate for your system.

For example:

- PyTorch Build: **Stable**
- Your OS: **Windows**
- Package: **Pip**
- Language: **Python**
- Compute Platform: **Newest CUDA version** (e.g., CUDA 12.4, ensuring it is not higher than your installed CUDA Toolkit version)

Then it will show a command next to _"Run this command:"_ like this:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If a GPU is not available, install the CPU build of the `torch` and `torchaudio` versions pinned in `requirements.txt`. With the current pins, the command is:

```bash
pip3 install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

Run the command in a terminal window.

### Diarization Assets (Hugging Face)

Speaker diarization requires assets that live behind gated repositories on Hugging Face. The recommended (and air-gapped-friendly) setup is:

1. Create an account on [Hugging Face](https://huggingface.co/).
2. Accept the licenses for `pyannote/speaker-diarization-community-1` and `pyannote/segmentation`.
3. Generate a fine-grained access token with read permissions on `pyannote/speaker-diarization-community-1`, `pyannote/segmentation`, and `speechbrain/spkrec-ecapa-voxceleb`.
4. Set the token on the machine that will clone the models:

   ```bash
   # On Windows:
   set HUGGING_FACE_HUB_TOKEN=your_token_here

   # On macOS/Linux:
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

5. Install [Git LFS](https://git-lfs.com/) (on macOS, run `brew install git-lfs`) and clone the pipelines directly into the project root, beside `app.py`. If prompted for a password, use the Hugging Face access token from above:

   ```bash
   git lfs install
   git clone https://hf.co/pyannote/speaker-diarization-community-1 pyannote-speaker-diarization-community-1
   ```

6. If you prepared the repository on another machine, copy the `pyannote-speaker-diarization-community-1` folder into the Queue Whisperer directory before starting the server.

Queue Whisperer automatically looks for this directory next to `app.py` and loads the pipeline directly from disk. If the directory is missing, the app falls back to downloading from Hugging Face using the configured token.

## Usage

1. **Start the Server:**
   ```bash
   python app.py
   ```
2. **Access the Application:**  
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
   By default the server only accepts connections from the same machine. See [Configuration](#configuration) and [Security](#security) before you change that.
3. **Submit a Job:**  
   Use the web interface to upload an audio or video file, select the transcription mode, and submit the job.
4. **Monitor Progress:**  
   The job queue system provides real-time status updates via WebSocket.
5. **Download Results:**  
   Once a job is complete, download the transcription with or without timestamps.

## HTTP API

Everything the web interface does is also available over HTTP:

| Method and path | Purpose |
|---|---|
| `POST /upload` | Multipart form with `file`, `mode` (`one`, `stereo`, `diarization`) and `language` (a Whisper language code, default `en`). Returns the job id. |
| `GET /jobs` | All jobs with status, progress and ETA as JSON. |
| `GET /download/{job_id}/without` | Plain transcript of a finished job. |
| `GET /download/{job_id}/with` | Transcript with `[start - end]` timestamps. |
| `GET /restart/{job_id}` | Queue a failed or cancelled job again. |
| `GET /remove/{job_id}` | Delete a job and its files. |
| `WS /ws` | Sends the job table as JSON once per second. |

The uploaded filename must be a plain name without a path and end in one of these extensions: 3gp, aac, aif, aiff, amr, avi, flac, m4a, m4v, mkv, mov, mp3, mp4, mpeg, mpg, oga, ogg, opus, wav, webm, wma, wmv. The server stores the file under a generated name inside `jobs/<job id>/` and keeps the original name for display and for the download filename.

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

Queue Whisperer has no user management. Anyone who can reach the port can upload files, list jobs and download transcripts. By default it therefore listens on the loopback interface only, so only programs on the same machine can reach it.

If you set `WHISPER_JOB_SERVER_TOKEN`, every request that creates, restarts or removes a job must carry `Authorization: Bearer <token>`. The web interface asks for the token the first time the server answers 401 and keeps it for the browser tab. The token does not cover `GET /jobs`, the downloads or the WebSocket, so it stops strangers from submitting work, not from reading results.

To expose the server beyond the machine, put it behind a reverse proxy that authenticates users and terminates TLS, and point the proxy at the loopback address. Do not set `WHISPER_JOB_SERVER_HOST=0.0.0.0` on a network you do not control.

Uploaded files are stored under a server-generated name, never under the name the client sent, and uploads are capped by `WHISPER_JOB_SERVER_MAX_UPLOAD_MB`.

## Example

`examples/tone.wav` is a one-second 440 Hz tone, 16 kHz mono, written by `examples/make_wav.py` with the standard library `wave` module. It contains no speech, so do not expect a real transcript; on a CPU-only run Whisper wrote `Beep.` for it. It runs the whole path: upload, FFmpeg conversion, model load, transcription and download.

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

When `GET /jobs` reports `"status":"done"`, fetch the transcripts:

```bash
curl -OJ http://127.0.0.1:8000/download/<id>/without   # tone_transcription.txt
curl -OJ http://127.0.0.1:8000/download/<id>/with      # tone_transcription_timestamps.txt
```

With a token set, add `-H "Authorization: Bearer <token>"` to the upload call.

To regenerate the example file:

```bash
python3 examples/make_wav.py
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
node --test tests/test_ui.cjs
```

The Python tests use FastAPI's `TestClient` and replace `torch`, `torchaudio` and `whisper` with small fakes, so they need neither the ML packages nor a model download. They cover the filename and extension checks, the upload limit, the token, the empty job list, and worker runs on synthetic mono and stereo audio. The worker tests need `ffmpeg` and `ffprobe` on the PATH and are skipped otherwise. The JavaScript tests check the inline UI renderer using Node.js 24, with no npm dependencies. CI runs both commands on Ubuntu with Python 3.12 and Node.js 24.

## Job Management

- **Real-Time Updates:** Job status and progress are communicated via WebSocket.
- **Queue Processing:** Transcription jobs are queued and processed sequentially.
- **Job Removal:** Completed or failed jobs can be removed through the provided endpoints.

## Troubleshooting

- **FFmpeg Issues:** Verify FFmpeg is installed and available in your PATH.
- **Initial Model Download:** The first run may take extra time to download the Whisper large-v2 model (approximately 3GB).
- **CUDA Errors:** Ensure your GPU is CUDA-capable and that the correct PyTorch version is installed; use `nvidia-smi` to verify GPU status.
- **Hugging Face Token:** Make sure your Hugging Face token is set correctly for speaker detection.
- **Job Failures:** Check the logs in the `logs` directory for more detailed error messages.

## Notes

- **Resource Usage:** Transcription, particularly in speaker detection mode, can be resource intensive.
- **Model Caching:** The Whisper model is cached after the initial download.
- **Customization:** Built with FastAPI, Queue Whisperer is designed to be easily extended or modified to meet your requirements.
- **Model Loading:** The model is fixed to `large-v2` and is loaded again for every job; a stereo job loads it three times.
- **HTTPS Proxies:** The web interface opens its WebSocket with `ws://`, so live updates do not work when the page is served over HTTPS through a proxy. Polling `GET /jobs` still works.
- **Languages:** The web interface offers English and German; the API accepts any Whisper language code.

## License

MIT, see [LICENSE](LICENSE).
