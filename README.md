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
  - [Hugging Face Authentication](#hugging-face-authentication)
- [Usage](#usage)
- [Job Management](#job-management)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

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

- **Python:** Version 3.10.0 up to (but not including) 3.14. Any 3.13.x release is supported, but Python 3.14+ is not.
- **FFmpeg:** Installed and available in your system PATH.
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

If a GPU is not available, choose the CPU version under "Compute Platform", where the command may look like this:

```bash
pip3 install torch torchvision torchaudio
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

5. Install [Git LFS](https://git-lfs.com/, on macOS, run `brew install git-lfs`) and clone the pipelines directly into the project root (beside `app.py`, if prompted for a password, use a Huggingface access token from above):

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
3. **Submit a Job:**  
   Use the web interface to upload an audio or video file, select the transcription mode, and submit the job.
4. **Monitor Progress:**  
   The job queue system provides real-time status updates via WebSocket.
5. **Download Results:**  
   Once a job is complete, download the transcription with or without timestamps.

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
