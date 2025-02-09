# Audio Transcription Service

A web application that provides audio transcription services with support for single speaker, stereo (two-speaker), and speaker diarization modes. This documentation outlines installation, configuration, usage, and feature details to help you get started quickly.

## Prerequisites

- **Python:** Version 3.8 or higher (ensure pip is installed)
- **FFmpeg:** Installed on your system
- **GPU:** CUDA-capable GPU is recommended for faster processing (CPU mode is supported)
- **NVIDIA Software:**
  - [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  - [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
  - Use `nvidia-smi` to verify that your GPU is recognized and compatible.
- **Hugging Face Account:** Required for speaker diarization mode

## Installation

### 1. Install FFmpeg
- **Windows:** 
  - Download from the [FFmpeg website](https://ffmpeg.org/download.html) and add it to your system PATH, or install via Chocolatey:
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

### 2. Set Up a Virtual Environment
Navigate to your project directory, then create and activate a virtual environment:
```bash
python -m venv whisper_env

# Activate the virtual environment:
# On Windows:
whisper_env\Scripts\activate
# On macOS/Linux:
source whisper_env/bin/activate
```

### 3. Install Required Python Packages
Install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA Support
If you have a CUDA-capable GPU, install PyTorch with CUDA support. Visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page and select:
- **Stable**
- **Windows** (or your respective OS)
- **Pip**
- **Python**
- **Newest CUDA version** (e.g., CUDA 12.4, ensuring it is not higher than your installed CUDA Toolkit version)

For example, for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If you do not have a compatible GPU, install the CPU version:
```bash
pip install torch torchvision torchaudio
```

### 5. Configure Hugging Face Authentication
For speaker detection, obtain a Hugging Face authentication token:
- Create an account on [Hugging Face](https://huggingface.co/)
- Accept the license agreement for `pyannote/speaker-diarization`
- Create an access token at [Hugging Face Settings](https://huggingface.co/settings/tokens)
- Set the token as an environment variable:
  ```bash
  # On Windows:
  set HUGGING_FACE_HUB_TOKEN=your_token_here

  # On macOS/Linux:
  export HUGGING_FACE_HUB_TOKEN=your_token_here
  ```

## Usage

### 1. Start the Server
Run the application server:
```bash
python app.py
```

### 2. Access the Application
Open your web browser and navigate to:
```
http://localhost:8000
```

### 3. Transcription Process
- **Upload an Audio File:** Use the upload button to select your audio file.
- **Select Transcription Mode:**
    - **One Speaker Mode:** For files with a single speaker.
    - **Stereo Mode:** For files with two speakers on separate audio channels.
    - **Speaker Detection Mode:** For files with multiple speakers (AI-powered detection).
- **Monitor Progress:** Real-time updates are provided via WebSocket.
- **Download Results:** Once processing is complete, download the transcribed text.

## Features

- **Real-Time Progress Updates:** Live status updates via WebSocket.
- **Multiple Transcription Modes:** Supports one speaker, stereo, and speaker detection.
- **Timestamp Options:** Output transcriptions with or without timestamps.
- **Job Management:** View job status, download results, and remove completed jobs.
- **Format Support:** Handles various audio and video file formats.

## Notes

- The initial run will download the Whisper large-v2 model (~3GB).
- Speaker Detection Mode requires additional processing time and resources.
- For optimal performance, a CUDA-capable GPU is recommended.
- Ensure sufficient disk space and a stable internet connection for model downloads.

## Troubleshooting

- **FFmpeg Not Found:** Verify that FFmpeg is installed and added to your system PATH.
- **CUDA Issues:** Confirm that your GPU is CUDA-capable, and that the correct version of PyTorch is installed.
- **GPU Recognition:** Use `nvidia-smi` to ensure your GPU is recognized.
- **Authentication Errors:** Double-check your Hugging Face token configuration if you encounter issues with speaker detection.
