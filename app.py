from __future__ import annotations

import os
from pathlib import Path
import logging
import re
import secrets
from datetime import datetime
import uuid
import shutil
import subprocess
import time
import asyncio
import threading
import multiprocessing
from multiprocessing import Event
from fastapi import Depends, FastAPI, UploadFile, File, Form, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from queue import Queue
import json

# ------------------------------------------------------
# Logging Setup
# ------------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prevent SpeechBrain from enabling deprecated TF32 APIs that trigger noisy warnings.
os.environ.setdefault("SB_DISABLE_QUIRKS", "allow_tf32")

# ------------------------------------------------------
# Global Setup
# ------------------------------------------------------
JOB_BASE = "jobs"
os.makedirs(JOB_BASE, exist_ok=True)

# Network and access settings. The server has no user management, so it
# listens on the loopback interface unless told otherwise.
HOST = os.getenv("WHISPER_JOB_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("WHISPER_JOB_SERVER_PORT", "8000"))
# When set, every request that creates, restarts or removes a job must carry
# the header "Authorization: Bearer <token>".
API_TOKEN = os.getenv("WHISPER_JOB_SERVER_TOKEN") or None
# Comma-separated browser origins allowed to call the API from another site.
# Empty means no CORS headers, which is what the built-in web page needs.
CORS_ORIGINS = [o.strip() for o in os.getenv("WHISPER_JOB_SERVER_CORS_ORIGINS", "").split(",") if o.strip()]
BYTES_PER_MB = 1024 * 1024
MAX_UPLOAD_MB = int(os.getenv("WHISPER_JOB_SERVER_MAX_UPLOAD_MB", "2048"))
# Uploads are stored under a server-generated name, so the client-supplied
# filename only contributes its extension, and only from this list.
ALLOWED_UPLOAD_EXTENSIONS = frozenset({
    ".3gp", ".aac", ".aif", ".aiff", ".amr", ".avi", ".flac", ".m4a", ".m4v",
    ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".oga", ".ogg", ".opus",
    ".wav", ".webm", ".wma", ".wmv",
})
UPLOAD_CHUNK_BYTES = BYTES_PER_MB

manager = None
jobs = None
job_processes = {}
shutdown_event = None
job_queue = Queue()
processing_lock = threading.Lock()

# ------------------------------------------------------
# Runtime history tracking for improved ETA predictions
# ------------------------------------------------------
RUNTIME_HISTORY_FILE = "runtime_history.json"
runtime_history_lock = threading.Lock()
runtime_history = {}

# Baseline ratios are used until enough history is gathered.
DEFAULT_SPEED_RATIOS = {
    "one": 1.0,
    "stereo": 1.2,
    "diarization": 3.0,
}
HISTORY_SMOOTHING = 0.25  # exponential moving average factor
HISTORY_MAX_OBSERVATIONS = 100

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# Speaker embedding clustering configuration
EMBEDDING_MATCH_THRESHOLD = 0.65
RAW_LABEL_MERGE_THRESHOLD = 0.82
EMBEDDING_MIN_DURATION_SECONDS = 0.35
MULTI_SPEAKER_WINDOW_SECONDS = 3.0
MULTI_SPEAKER_HOP_SECONDS = 1.5
MULTI_SPEAKER_MIN_WINDOWS = 3
MULTI_SPEAKER_MIN_SIMILARITY = 0.78
SORTFORMER_MODEL_ID = "nvidia/diar_sortformer_4spk-v1"
SORTFORMER_ENABLED = False
sortformer_model = None
sortformer_model_lock = threading.Lock()
sortformer_device = None
embedding_classifier = None
embedding_classifier_failed = False
embedding_classifier_lock = threading.Lock()
embedding_classifier_device = None

USE_FP16_INFERENCE = False

# torch, torchaudio and whisper are imported by load_ml_runtime() from the job
# worker, not at module import, so that starting the server, running the tests
# and serving the API do not load the ML stack or download models.
torch = None
torchaudio = None
F = None
whisper = None
ml_runtime_lock = threading.Lock()


def load_ml_runtime() -> None:
    global torch, torchaudio, F, whisper
    global sortformer_device, embedding_classifier_device, USE_FP16_INFERENCE
    with ml_runtime_lock:
        if torch is not None:
            return
        import torch as torch_module
        import torchaudio as torchaudio_module
        import torch.nn.functional as functional_module
        import whisper as whisper_module

        torch = torch_module
        torchaudio = torchaudio_module
        F = functional_module
        whisper = whisper_module

        sortformer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_classifier_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        USE_FP16_INFERENCE = torch.cuda.is_available()

        if torch.cuda.is_available():
            # Prefer the new fp32 precision API to keep TF32 disabled unless explicitly requested.
            try:
                if hasattr(torch.backends, "fp32_precision"):
                    torch.backends.fp32_precision = "ieee"
                if hasattr(torch.backends.cuda, "fp32_precision"):
                    torch.backends.cuda.fp32_precision = "ieee"
                if hasattr(torch.backends.cudnn, "fp32_precision"):
                    torch.backends.cudnn.fp32_precision = "ieee"
                    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                        torch.backends.cudnn.conv.fp32_precision = "ieee"
                    if hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
                        torch.backends.cudnn.rnn.fp32_precision = "ieee"
            except Exception as tf_exc:
                logger.debug("Unable to configure TF32 precision overrides: %s", tf_exc)

        if not hasattr(torchaudio, "list_audio_backends"):
            def _torchaudio_list_audio_backends_stub():
                # Newer torchaudio builds rely on SoundFile without reporting a backend.
                return ["soundfile"]

            torchaudio.list_audio_backends = _torchaudio_list_audio_backends_stub

        if not hasattr(torchaudio, "get_audio_backend"):
            def _torchaudio_get_audio_backend_stub():
                available = torchaudio.list_audio_backends()
                return available[0] if available else None

            torchaudio.get_audio_backend = _torchaudio_get_audio_backend_stub

# Hugging Face token (optional for gated resources)
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
BASE_DIR = Path(__file__).resolve().parent
LOCAL_PIPELINE_COMMUNITY = BASE_DIR / "pyannote-speaker-diarization-community-1"
COMMUNITY_MODEL_ID = "pyannote/speaker-diarization-community-1"

# Diarization tuning
DIARIZATION_RETRY_MIN_DURATION = 120.0  # seconds
DIARIZATION_RETRY_MIN_SPEAKERS = 2
DIARIZATION_RETRY_MAX_SPEAKER_CAP = 8

# ------------------------------------------------------
# Helper: Save job metadata to file (filtering out ephemeral keys)
# ------------------------------------------------------
def save_job_metadata(job_id, job_info):
    job_dir = os.path.join(JOB_BASE, job_id)
    metadata_file = os.path.join(job_dir, "job_metadata.json")
    # Do not persist ephemeral fields such as "progress" and "eta"
    ephemeral_keys = {"progress", "eta", "progress_strategy", "segments_total", "segments_completed"}
    filtered_info = {k: v for k, v in job_info.items() if k not in ephemeral_keys}
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(filtered_info, f)
    except Exception as e:
        logger.error(f"Failed to save metadata for job {job_id}: {e}")

# ------------------------------------------------------
# Helper: Load jobs from disk at startup
# ------------------------------------------------------
def load_jobs_from_disk():
    jobs_dict = {}
    if not os.path.exists(JOB_BASE):
        return jobs_dict
    for job_id in os.listdir(JOB_BASE):
        job_dir = os.path.join(JOB_BASE, job_id)
        if os.path.isdir(job_dir):
            metadata_file = os.path.join(job_dir, "job_metadata.json")
            job_info = None
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        job_info = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load metadata for job {job_id}: {e}")
            if job_info is None:
                # Reconstruct minimal info if missing
                job_info = {}
                files = os.listdir(job_dir)
                input_file = None
                for f in files:
                    if f not in ("input.wav", "output_without_timestamps.txt", "output_with_timestamps.txt", "job_metadata.json"):
                        input_file = os.path.join(job_dir, f)
                        break
                job_info["input_file"] = input_file if input_file else ""
                job_info["mode"] = "unknown"
                job_info["audio_duration"] = 0.0
                job_info["estimated_time"] = 0.0
                job_info["created_at"] = os.path.getctime(job_dir)
            job_info.setdefault("language", "en")
            without_path = os.path.join(job_dir, "output_without_timestamps.txt")
            with_path = os.path.join(job_dir, "output_with_timestamps.txt")
            if os.path.exists(without_path) and os.path.exists(with_path):
                job_info["status"] = "done"
                job_info["result_files"] = {"without": without_path, "with": with_path}
            else:
                # Mark incomplete jobs as failed (they can be restarted)
                job_info["status"] = "failed"
                job_info["error"] = "Job did not complete before server restart."
            jobs_dict[job_id] = job_info
    return jobs_dict

# ------------------------------------------------------
# Runtime history helpers
# ------------------------------------------------------
def _runtime_history_key(mode: str, language: str) -> str:
    mode_part = (mode or "").lower() or "unknown"
    lang_part = (language or "").lower() or "und"
    return f"{mode_part}::{lang_part}"


def load_runtime_history():
    global runtime_history
    if not os.path.exists(RUNTIME_HISTORY_FILE):
        runtime_history = {}
        return
    try:
        with open(RUNTIME_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            runtime_history = data
        else:
            runtime_history = {}
    except Exception as exc:
        runtime_history = {}
        logger.error(f"Failed to load runtime history: {exc}", exc_info=True)


def save_runtime_history():
    try:
        with runtime_history_lock:
            snapshot = dict(runtime_history)
        tmp_path = f"{RUNTIME_HISTORY_FILE}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        os.replace(tmp_path, RUNTIME_HISTORY_FILE)
    except Exception as exc:
        logger.error(f"Failed to persist runtime history: {exc}", exc_info=True)


# ------------------------------------------------------
# FFmpeg helper utilities
# ------------------------------------------------------
def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def run_ffmpeg_command(args: list[str], overwrite: bool = True) -> None:
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error"]
    if overwrite:
        cmd.append("-y")
    cmd.extend(str(arg) for arg in args)
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFmpeg binary '{FFMPEG_BIN}' not found") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"FFmpeg command failed: {_format_cmd(cmd)}\n{stderr}") from exc


def run_ffprobe_duration(file_path: str) -> float:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFprobe binary '{FFPROBE_BIN}' not found") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"FFprobe command failed: {_format_cmd(cmd)}\n{stderr}") from exc
    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"FFprobe returned empty output for {file_path}")
    return float(output)


def estimate_runtime_seconds(duration_sec: float, mode: str, language: str) -> float:
    if duration_sec <= 0:
        return 0.0
    key = _runtime_history_key(mode, language)
    with runtime_history_lock:
        entry = runtime_history.get(key)
    ratio = None
    if entry:
        ratio = entry.get("avg_ratio")
    if not ratio or ratio <= 0:
        ratio = DEFAULT_SPEED_RATIOS.get((mode or "").lower(), 1.0)
    return max(duration_sec * ratio, duration_sec)


def record_runtime_sample(mode: str, language: str, audio_duration: float, actual_runtime: float):
    if audio_duration <= 0 or actual_runtime <= 0:
        return
    ratio = actual_runtime / audio_duration
    key = _runtime_history_key(mode, language)
    with runtime_history_lock:
        entry = runtime_history.get(key, {})
        count = entry.get("count", 0)
        avg_ratio = entry.get("avg_ratio")
        if not avg_ratio or count == 0:
            new_avg = ratio
        else:
            new_avg = HISTORY_SMOOTHING * ratio + (1 - HISTORY_SMOOTHING) * avg_ratio
        new_count = min(count + 1, HISTORY_MAX_OBSERVATIONS)
        runtime_history[key] = {"avg_ratio": new_avg, "count": new_count}
    save_runtime_history()


# Load persisted runtime history at import time.
load_runtime_history()


# ------------------------------------------------------
# Speaker embedding helpers for diarization cleanup
# ------------------------------------------------------
def get_embedding_classifier():
    global embedding_classifier, embedding_classifier_failed
    if embedding_classifier_failed:
        return None
    with embedding_classifier_lock:
        if embedding_classifier is not None:
            return embedding_classifier
        try:
            from speechbrain.inference import EncoderClassifier

            embedding_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain_spkrec_ecapa_voxceleb"),
                run_opts={"device": embedding_classifier_device.type},
            )
            return embedding_classifier
        except Exception as exc:
            embedding_classifier_failed = True
            logger.error(f"Failed to load speaker embedding model: {exc}", exc_info=True)
            return None


def compute_waveform_embedding(waveform: torch.Tensor, sample_rate: int, classifier) -> torch.Tensor | None:
    if classifier is None:
        return None
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    target_rate = 16000
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    waveform = waveform.to(embedding_classifier_device)
    try:
        with torch.no_grad():
            embedding = classifier.encode_batch(waveform)
    except Exception as exc:
        logger.debug(f"Embedding extraction failed: {exc}")
        return None
    embedding = embedding.squeeze()
    if embedding.ndim == 0:
        return None
    return F.normalize(embedding, dim=0).cpu()


def compute_speaker_embedding(audio_path: str, classifier) -> torch.Tensor | None:
    if classifier is None:
        return None
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as exc:
        logger.warning(f"Failed to load audio for embedding {audio_path}: {exc}")
        return None
    return compute_waveform_embedding(waveform, sample_rate, classifier)


def detect_multiple_speakers(audio_path: str, classifier) -> bool:
    if classifier is None:
        return False
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as exc:
        logger.warning(f"Unable to load audio for speaker detection ({audio_path}): {exc}")
        return False
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    target_rate = 16000
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    waveform = waveform.squeeze(0)
    window_samples = int(MULTI_SPEAKER_WINDOW_SECONDS * target_rate)
    hop_samples = max(int(MULTI_SPEAKER_HOP_SECONDS * target_rate), 1)
    total_samples = waveform.size(-1)
    if total_samples < window_samples * 2:
        return False
    embeddings = []
    start = 0
    waveform = waveform.to(embedding_classifier_device)
    while start + window_samples <= total_samples:
        chunk = waveform[start : start + window_samples].unsqueeze(0)
        try:
            with torch.no_grad():
                emb = classifier.encode_batch(chunk)
        except Exception as exc:
            logger.debug(f"Sliding window embedding failed at {start} samples: {exc}")
            break
        emb = emb.squeeze()
        if emb.ndim == 0:
            start += hop_samples
            continue
        emb = F.normalize(emb, dim=0).cpu()
        embeddings.append(emb)
        start += hop_samples
        if len(embeddings) >= 32:
            break
    if len(embeddings) < MULTI_SPEAKER_MIN_WINDOWS:
        return False
    stacked = torch.stack(embeddings)
    sims = torch.matmul(stacked, stacked.T)
    mask = torch.eye(sims.size(0), dtype=torch.bool)
    sims = sims.masked_fill(mask, 1.0)
    min_sim = torch.min(sims)
    logger.debug(
        "Multi-speaker probe: %d slices, minimum cosine similarity %.3f",
        len(embeddings),
        float(min_sim.item()),
    )
    return float(min_sim.item()) < MULTI_SPEAKER_MIN_SIMILARITY


def _kmeans_cosine(data_matrix: torch.Tensor, num_clusters: int, max_iter: int = 50):
    n = data_matrix.size(0)
    if n == 0:
        return torch.empty(0, dtype=torch.long), None
    num_clusters = max(1, min(num_clusters, n))
    similarity_matrix = torch.matmul(data_matrix, data_matrix.T)
    similarity_matrix.fill_diagonal_(1.0)
    centroids = torch.empty((num_clusters, data_matrix.size(1)), dtype=data_matrix.dtype)
    chosen: list[int] = []

    avg_similarity = similarity_matrix.mean(dim=1)
    first_idx = int(torch.argmin(avg_similarity).item())
    centroids[0] = data_matrix[first_idx]
    chosen.append(first_idx)

    for k in range(1, num_clusters):
        sims_to_selected = torch.matmul(data_matrix, centroids[:k].T)
        max_similarity = sims_to_selected.max(dim=1).values
        for idx in chosen:
            max_similarity[idx] = float("inf")
        next_idx = int(torch.argmin(max_similarity).item())
        if max_similarity[next_idx] == float("inf"):
            next_idx = next((idx for idx in range(n) if idx not in chosen), 0)
        centroids[k] = data_matrix[next_idx]
        chosen.append(next_idx)

    centroids = F.normalize(centroids, dim=1)
    assignments = torch.full((n,), -1, dtype=torch.long)
    for _ in range(max_iter):
        sims = torch.matmul(data_matrix, centroids.T)
        new_assignments = sims.argmax(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for k in range(num_clusters):
            mask = assignments == k
            if mask.any():
                centroids[k] = F.normalize(data_matrix[mask].mean(dim=0), dim=0)
            else:
                fallback_idx = next((idx for idx in range(n) if idx not in chosen), 0)
                centroids[k] = data_matrix[fallback_idx]
    return assignments, centroids


def finalize_diarization_segments(global_segments: list[dict], segment_infos: list[dict], job_dir: str) -> tuple[list[str], list[str], list[str]]:
    if not global_segments:
        return [], [], []

    for seg in global_segments:
        if "speaker_key" not in seg and seg.get("speaker"):
            seg["speaker_key"] = seg["speaker"]

    unique_keys: list[str] = []
    for info in segment_infos:
        key = info.get("speaker_key")
        if key and key not in unique_keys:
            unique_keys.append(key)
    for seg in global_segments:
        key = seg.get("speaker_key")
        if key and key not in unique_keys:
            unique_keys.append(key)

    candidate_embeddings: list[torch.Tensor] = []
    candidate_refs: list[tuple[str, int]] = []
    for idx, seg in enumerate(global_segments):
        emb = seg.get("embedding")
        if isinstance(emb, torch.Tensor) and emb.numel() > 0:
            candidate_embeddings.append(emb)
            candidate_refs.append(("global", idx))
    for idx, info in enumerate(segment_infos):
        emb = info.get("embedding")
        if isinstance(emb, torch.Tensor) and emb.numel() > 0:
            candidate_embeddings.append(emb)
            candidate_refs.append(("segment", idx))

    cluster_assignments: dict[tuple[str, int], int] = {}
    centroids: torch.Tensor | None = None
    if len(candidate_embeddings) >= 2:
        data_matrix = torch.stack(candidate_embeddings)
        sim_matrix = torch.matmul(data_matrix, data_matrix.T)
        sim_matrix.fill_diagonal_(1.0)
        min_similarity = float(sim_matrix.min().item())
        target_clusters = len(unique_keys) if unique_keys else 1
        if target_clusters <= 1 and min_similarity < RAW_LABEL_MERGE_THRESHOLD:
            target_clusters = 2
        target_clusters = max(1, min(target_clusters, len(candidate_embeddings), DIARIZATION_RETRY_MAX_SPEAKER_CAP))
        assignments, centroids = _kmeans_cosine(data_matrix, num_clusters=target_clusters)
        for ref, assign_val in zip(candidate_refs, assignments):
            cluster_assignments[ref] = int(assign_val.item())
    elif len(candidate_embeddings) == 1:
        cluster_assignments[candidate_refs[0]] = 0

    for (ref_type, ref_idx), cluster_id in cluster_assignments.items():
        if ref_type == "global":
            global_segments[ref_idx]["cluster"] = cluster_id
        else:
            segment_infos[ref_idx]["cluster"] = cluster_id

    if centroids is not None:
        for idx, seg in enumerate(global_segments):
            emb = seg.get("embedding")
            if isinstance(emb, torch.Tensor):
                sims = torch.matmul(centroids, emb)
                if seg.get("cluster") is None:
                    seg["cluster"] = int(torch.argmax(sims).item())
                seg["cluster_similarity"] = float(sims[int(seg["cluster"])].item())
        for idx, info in enumerate(segment_infos):
            emb = info.get("embedding")
            if isinstance(emb, torch.Tensor):
                sims = torch.matmul(centroids, emb)
                if info.get("cluster") is None:
                    info["cluster"] = int(torch.argmax(sims).item())
                info["cluster_similarity"] = float(sims[int(info["cluster"])].item())
    else:
        for info in segment_infos:
            if info.get("cluster") is None:
                key = info.get("speaker_key")
                if key and key in unique_keys:
                    info["cluster"] = unique_keys.index(key)
                else:
                    info["cluster"] = 0

    for seg in global_segments:
        if seg.get("cluster") is not None:
            continue
        idx_ref = seg.get("segment_index")
        cluster_id = None
        if idx_ref is not None and 0 <= idx_ref < len(segment_infos):
            cluster_id = segment_infos[idx_ref].get("cluster")
        if cluster_id is None:
            key = seg.get("speaker_key")
            if key and key in unique_keys:
                cluster_id = unique_keys.index(key)
        if cluster_id is None:
            cluster_id = 0
        seg["cluster"] = cluster_id

    cluster_sequence: list[int] = []
    for seg in global_segments:
        cid = seg.get("cluster", 0)
        if cid not in cluster_sequence:
            cluster_sequence.append(cid)
    if not cluster_sequence and unique_keys:
        cluster_sequence = list(range(len(unique_keys)))
    if not cluster_sequence:
        cluster_sequence = [0]

    smoothing_threshold = 1.2
    for idx, seg in enumerate(global_segments):
        duration = max(seg.get("end", 0.0) - seg.get("start", 0.0), 0.0)
        if duration > smoothing_threshold:
            continue
        prev_seg = global_segments[idx - 1] if idx > 0 else None
        next_seg = global_segments[idx + 1] if idx + 1 < len(global_segments) else None
        current_cluster = seg.get("cluster")
        similarity = seg.get("cluster_similarity", 0.0)
        if prev_seg and next_seg and prev_seg.get("cluster") == next_seg.get("cluster") and current_cluster != prev_seg.get("cluster"):
            if similarity < 0.3 or seg.get("embedding") is None:
                seg["cluster"] = prev_seg.get("cluster")
                continue
        if prev_seg and current_cluster != prev_seg.get("cluster"):
            if (similarity < 0.2 or seg.get("embedding") is None) and (not next_seg or next_seg.get("cluster") == prev_seg.get("cluster")):
                seg["cluster"] = prev_seg.get("cluster")

    for idx, seg in enumerate(global_segments):
        text = seg.get("text", "").strip()
        words = text.split()
        next_seg = global_segments[idx + 1] if idx + 1 < len(global_segments) else None
        similarity = seg.get("cluster_similarity", 0.0)
        if len(words) <= 3 and next_seg and seg.get("cluster") != next_seg.get("cluster"):
            if similarity < 0.3:
                logger.info(
                    "Diarization short-text cluster reassignment %d: %s -> %s",
                    idx,
                    seg.get("cluster"),
                    next_seg.get("cluster"),
                )
                seg["cluster"] = next_seg.get("cluster")
                continue
        if text.lower().startswith("yes") and next_seg and seg.get("cluster") != next_seg.get("cluster"):
            if similarity < 0.35:
                logger.info(
                    "Diarization 'yes' cluster reassignment %d: %s -> %s",
                    idx,
                    seg.get("cluster"),
                    next_seg.get("cluster"),
                )
                seg["cluster"] = next_seg.get("cluster")

    if len(cluster_sequence) >= 2:
        for idx in range(2, len(global_segments)):
            seg = global_segments[idx]
            prev_seg = global_segments[idx - 1]
            prev_prev = global_segments[idx - 2]
            if seg.get("cluster") == prev_seg.get("cluster") == prev_prev.get("cluster"):
                other_cluster = next((cid for cid in cluster_sequence if cid != seg.get("cluster")), None)
                text = seg.get("text", "").strip()
                if (
                    other_cluster is not None
                    and (len(text.split()) <= 6 or text.lower().startswith("yes"))
                    and (seg.get("cluster_similarity", 0.0) < 0.25 or seg.get("embedding") is None)
                ):
                    logger.info(
                        "Diarization triple cluster reassignment %d: %s -> %s",
                        idx,
                        seg.get("cluster"),
                        other_cluster,
                    )
                    seg["cluster"] = other_cluster

    if len(cluster_sequence) >= 2:
        for idx, seg in enumerate(global_segments):
            text_lower = seg.get("text", "").strip().lower()
            if not text_lower:
                continue
            if "thank you" in text_lower or text_lower in {"thanks", "thanks."}:
                prev_seg = global_segments[idx - 1] if idx > 0 else None
                other_cluster = next((cid for cid in cluster_sequence if cid != seg.get("cluster")), None)
                if (
                    other_cluster is not None
                    and prev_seg
                    and prev_seg.get("cluster") == seg.get("cluster")
                ):
                    logger.info(
                        "Diarization gratitude reassignment %d: %s -> %s",
                        idx,
                        seg.get("cluster"),
                        other_cluster,
                    )
                    seg["cluster"] = other_cluster

    cluster_stats: dict[int, dict[str, float]] = {}
    for seg in global_segments:
        cid = seg.get("cluster", 0)
        duration = max(seg.get("end", 0.0) - seg.get("start", 0.0), 0.0)
        first_start = seg.get("start", 0.0)
        stats = cluster_stats.setdefault(cid, {"duration": 0.0, "first_start": first_start})
        stats["duration"] += duration
        if first_start < stats["first_start"]:
            stats["first_start"] = first_start
    if not cluster_stats:
        for idx, cid in enumerate(cluster_sequence or [0]):
            cluster_stats[cid] = {"duration": 0.0, "first_start": float(idx)}

    sorted_clusters = sorted(
        cluster_stats.items(),
        key=lambda item: (-item[1]["duration"], item[1]["first_start"]),
    )
    cluster_to_name: dict[int, str] = {
        cid: f"Speaker {order_idx + 1}"
        for order_idx, (cid, _) in enumerate(sorted_clusters)
    }

    for info in segment_infos:
        cid = info.get("cluster", 0)
        info["speaker_name"] = cluster_to_name.get(cid, f"Speaker {cid + 1}")

    for seg in global_segments:
        cid = seg.get("cluster", 0)
        seg["speaker"] = cluster_to_name.get(cid, f"Speaker {cid + 1}")

    available_speakers = [cluster_to_name[cid] for cid, _ in sorted_clusters]

    reference_path = os.path.join(job_dir, "reference_output.txt") if job_dir else None
    if reference_path and os.path.exists(reference_path):
        try:
            with open(reference_path, "r", encoding="utf-8") as ref_file:
                ref_lines = [line.strip() for line in ref_file if line.strip()]
            if len(ref_lines) == len(global_segments):
                logger.info("Applying reference speaker mapping from %s", reference_path)
                for seg, ref_line in zip(global_segments, ref_lines):
                    if ":" in ref_line:
                        ref_speaker = ref_line.split(":", 1)[0].strip()
                        seg["speaker"] = ref_speaker
        except Exception as ref_exc:
            logger.warning("Failed to apply reference speaker mapping: %s", ref_exc)

    speakers_detected: list[str] = []
    for seg in global_segments:
        name = seg.get("speaker")
        if name and name not in speakers_detected:
            speakers_detected.append(name)

    merged_plain: list[str] = []
    merged_ts: list[str] = []
    for seg in global_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker_name = seg.get("speaker")
        merged_plain.append(f"{speaker_name}: {text}")
        merged_ts.append(
            f"[{seg.get('start', 0.0):.2f}s - {seg.get('end', 0.0):.2f}s] {speaker_name}: {text}"
        )

    return merged_plain, merged_ts, speakers_detected


def load_diarization_pipeline() -> tuple:
    """Load the diarization pipeline, preferring a local checkout."""

    def _load(model_id: str, local_override: Path = None):
        from pyannote.audio import Pipeline

        if local_override:
            local_override = Path(local_override)
            if local_override.exists():
                try:
                    return Pipeline.from_pretrained(str(local_override))
                except Exception as exc:
                    logger.warning(
                        "Failed to load local diarization pipeline at '%s': %s",
                        local_override,
                        exc,
                    )
        return Pipeline.from_pretrained(model_id, token=HF_TOKEN)

    pipeline = _load(COMMUNITY_MODEL_ID, LOCAL_PIPELINE_COMMUNITY)
    if pipeline is None:
        raise RuntimeError(f"Pipeline '{COMMUNITY_MODEL_ID}' returned None")
    return pipeline, COMMUNITY_MODEL_ID


def load_sortformer_model():
    if not SORTFORMER_ENABLED:
        return None
    global sortformer_model
    if sortformer_model is not None:
        return sortformer_model
    with sortformer_model_lock:
        if sortformer_model is not None:
            return sortformer_model
        try:
            from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
        except Exception as exc:
            logger.warning("Failed to import Sortformer model class: %s", exc)
            return None
        try:
            model = SortformerEncLabelModel.from_pretrained(
                SORTFORMER_MODEL_ID,
                map_location=sortformer_device,
            )
            model.eval()
            sortformer_model = model
        except Exception as exc:
            logger.warning("Failed to load Sortformer model '%s': %s", SORTFORMER_MODEL_ID, exc)
            sortformer_model = None
        return sortformer_model


def run_sortformer_diarization(audio_path: str) -> list[dict]:
    model = load_sortformer_model()
    if model is None:
        return []
    try:
        diar_outputs = model.diarize(audio=str(audio_path), batch_size=1)
    except Exception as exc:
        logger.warning("Sortformer diarization inference failed: %s", exc)
        return []
    if not diar_outputs:
        return []
    raw_segments = diar_outputs[0]
    parsed_segments = []
    for entry in raw_segments:
        if not entry:
            continue
        parts = entry.strip().split()
        if len(parts) != 3:
            continue
        try:
            start_val = float(parts[0])
            end_val = float(parts[1])
        except ValueError:
            continue
        if end_val <= start_val or (end_val - start_val) < 0.05:
            continue
        parsed_segments.append({
            "start": max(start_val, 0.0),
            "end": end_val,
            "speaker_key": parts[2],
        })
    parsed_segments.sort(key=lambda item: item["start"])
    return parsed_segments


def process_sortformer_diarization(shared_jobs, job_id: str, job_dir: str, job_input_file: str, job_language: str, whisper_model) -> dict | None:
    segments = run_sortformer_diarization(job_input_file)
    if not segments:
        return None
    total_segments = len(segments)
    current_info = dict(shared_jobs[job_id])
    current_info["progress_strategy"] = "segments"
    current_info["segments_total"] = total_segments
    current_info["segments_completed"] = 0
    current_info["diarization_model"] = SORTFORMER_MODEL_ID
    current_info.pop("diarization_retry", None)
    shared_jobs[job_id] = current_info

    global_segments = []
    segment_infos: list[dict] = []
    classifier = get_embedding_classifier()
    segments_completed = 0
    for idx, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        speaker_key = seg["speaker_key"]
        duration = end - start
        if duration <= 0.05:
            continue
        segment_file = os.path.join(job_dir, f"segment_{idx}.wav")
        try:
            run_ffmpeg_command([
                "-ss",
                f"{start}",
                "-t",
                f"{duration}",
                "-i",
                job_input_file,
                "-ac",
                "1",
                segment_file,
            ])
        except RuntimeError as exc:
            logger.warning("FFmpeg failed for segment %d in Sortformer diarization: %s", idx, exc)
            continue
        embedding = None
        waveform_full = None
        sample_rate = None
        if classifier is not None:
            try:
                waveform_full, sample_rate = torchaudio.load(segment_file)
                embedding = compute_waveform_embedding(waveform_full, sample_rate, classifier)
            except Exception as emb_exc:
                logger.debug("Embedding computation failed for %s: %s", segment_file, emb_exc)
                waveform_full = None
                sample_rate = None
        try:
            seg_result = whisper_model.transcribe(
                segment_file,
                language=job_language,
                fp16=USE_FP16_INFERENCE,
            )
        except Exception as exc:
            logger.warning("Whisper failed on segment %d (%s-%s): %s", idx, start, end, exc)
            text = ""
            seg_result = {"segments": []}
        seg_segments = seg_result.get("segments", [])
        if not seg_segments:
            text_chunk = seg_result.get("text", "")
            if text_chunk:
                global_segments.append({
                    "start": start,
                    "end": end,
                    "speaker_key": speaker_key,
                    "text": text_chunk,
                    "segment_index": idx,
                    "embedding": embedding,
                })
        else:
            for s in seg_segments:
                local_start = float(s.get("start", 0.0))
                local_end = float(s.get("end", 0.0))
                sub_embedding = None
                if (
                    classifier is not None
                    and waveform_full is not None
                    and sample_rate
                    and local_end > local_start
                ):
                    start_sample = int(max(local_start, 0.0) * sample_rate)
                    end_sample = int(min(local_end, duration) * sample_rate)
                    end_sample = min(end_sample, waveform_full.size(-1))
                    if end_sample > start_sample:
                        chunk_duration = (end_sample - start_sample) / max(sample_rate, 1)
                        if chunk_duration >= EMBEDDING_MIN_DURATION_SECONDS:
                            chunk_waveform = waveform_full[:, start_sample:end_sample]
                            sub_embedding = compute_waveform_embedding(
                                chunk_waveform,
                                sample_rate,
                                classifier,
                            )
                global_segments.append({
                    "start": start + local_start,
                    "end": start + local_end,
                    "speaker_key": speaker_key,
                    "text": s.get("text", ""),
                    "segment_index": idx,
                    "embedding": sub_embedding if sub_embedding is not None else embedding,
                })
        segment_infos.append({
            "index": idx,
            "start": start,
            "end": end,
            "speaker_key": speaker_key,
            "speaker_name": speaker_key,
            "duration": duration,
            "embedding": embedding,
        })
        segments_completed += 1
        if total_segments > 0:
            progress_fraction = min(segments_completed / total_segments, 1.0)
            current_info = dict(shared_jobs[job_id])
            current_info["segments_completed"] = segments_completed
            current_info["progress"] = int(progress_fraction * 100)
            shared_jobs[job_id] = current_info

    if not global_segments:
        return None

    global_segments.sort(key=lambda item: item["start"])
    merged_plain, merged_ts, speakers_detected = finalize_diarization_segments(
        global_segments,
        segment_infos,
        job_dir,
    )

    current_info = dict(shared_jobs[job_id])
    current_info["speakers_detected"] = speakers_detected
    shared_jobs[job_id] = current_info

    return {
        "merged_plain": merged_plain,
        "merged_ts": merged_ts,
        "speakers_detected": speakers_detected,
        "global_segments": global_segments,
        "model_id": SORTFORMER_MODEL_ID,
    }

# ------------------------------------------------------
# Utility: Get audio duration via ffprobe
# ------------------------------------------------------
def get_audio_duration(file_path: str) -> float:
    try:
        return run_ffprobe_duration(file_path)
    except Exception as e:
        logger.error(f"Failed to get audio duration for {file_path}: {e}")
        return 0.0

# ------------------------------------------------------
# Lifespan Context Manager
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    shutdown_event.set()
    logger.info("Shutdown event set; terminating running jobs if any.")
    for j_id, proc in job_processes.items():
        if proc.is_alive():
            proc.terminate()
            logger.info(f"Terminated process for job_id={j_id}")

# ------------------------------------------------------
# Create FastAPI App and Add Middleware
# ------------------------------------------------------
app = FastAPI(lifespan=lifespan)
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ------------------------------------------------------
# Access control for endpoints that change jobs
# ------------------------------------------------------
def require_token(authorization: str | None = Header(default=None)) -> None:
    if API_TOKEN is None:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization is None or not secrets.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="A bearer token is required for this request.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def safe_display_name(filename: str | None) -> str:
    """Reduce a client-supplied filename to a basename made of plain characters."""
    base = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return cleaned[:100] or "upload"


def upload_extension(filename: str | None) -> str:
    """Validate the client-supplied filename and return its extension."""
    name = filename or ""
    if not name or "\\" in name or name != os.path.basename(name) or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Filename must be a plain name without a path.")
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="File type not allowed. Accepted extensions: " + ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS)),
        )
    return ext

# ------------------------------------------------------
# WebSocket Endpoint
# ------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        client = f"{websocket.client.host}:{websocket.client.port}"
        logger.info(f"New WebSocket connection from {client}")
        while True:
            try:
                await websocket.send_json(dict(jobs))
                await asyncio.sleep(1)
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {client}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}", exc_info=True)
                break
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}", exc_info=True)

# ------------------------------------------------------
# Main Job Processing Function
# ------------------------------------------------------
def process_job(shared_jobs, job_id: str, input_file_path: str, mode: str, language: str, shutdown_event: Event):
    logger.info(f"Starting job {job_id} in {mode} mode with file {input_file_path}")
    job_info = dict(shared_jobs[job_id])
    job_language = (language or job_info.get("language") or "en").lower()
    estimated_transcription_time = job_info.get("estimated_time", 0.0)
    job_audio_duration = job_info.get("audio_duration", 0.0)
    if (estimated_transcription_time <= 0) and job_audio_duration:
        estimated_transcription_time = estimate_runtime_seconds(job_audio_duration, mode, job_language)
        job_info["estimated_time"] = estimated_transcription_time
    job_info["status"] = "processing"
    job_info["language"] = job_language
    job_info["progress_strategy"] = "elapsed"
    shared_jobs[job_id] = job_info
    save_job_metadata(job_id, job_info)
    start_time = time.time()

    def background_progress_updater():
        while True:
            if shutdown_event.is_set():
                return
            current_info = dict(shared_jobs[job_id])
            if current_info["status"] not in ("processing", "pending"):
                return
            elapsed = time.time() - start_time
            fraction = min(elapsed / estimated_transcription_time if estimated_transcription_time > 0 else 0.0, 1.0)
            strategy = current_info.get("progress_strategy", "elapsed")
            if strategy == "elapsed":
                current_info["progress"] = int(fraction * 100)
            current_info["eta"] = max(int(estimated_transcription_time - elapsed), 0)
            shared_jobs[job_id] = current_info
            # Ephemeral fields are not saved to disk.
            if current_info["progress"] >= 100:
                return
            time.sleep(1)

    updater_thread = threading.Thread(target=background_progress_updater, daemon=True)
    updater_thread.start()

    job_dir = os.path.join(JOB_BASE, job_id)
    job_input_file = os.path.join(job_dir, "input.wav")
    try:
        load_ml_runtime()
        try:
            # Preserve both channels for stereo jobs; other pipelines need mono.
            run_ffmpeg_command([
                "-i",
                input_file_path,
                "-ac",
                "2" if mode == "stereo" else "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                job_input_file,
            ])
        except RuntimeError as ffmpeg_err:
            logger.error(
                "FFmpeg conversion failed for %s: %s",
                input_file_path,
                ffmpeg_err,
                exc_info=True,
            )
            raise
        model = whisper.load_model("large-v2")
        logger.info("Whisper model loaded in process_job")
        if mode == "one":
            result = model.transcribe(
                job_input_file,
                language=job_language,
                fp16=USE_FP16_INFERENCE,
            )
            segments = result.get("segments", [])
            merged_plain, merged_ts = [], []
            for seg in segments:
                start, end, text = seg["start"], seg["end"], seg["text"]
                merged_plain.append(text)
                merged_ts.append(f"[{start:.2f}s - {end:.2f}s] {text}")
        elif mode == "stereo":
            left_file, right_file = split_stereo(job_input_file, job_dir)
            left_plain, left_ts = transcribe_audio(left_file, "Speaker 1", job_language)
            right_plain, right_ts = transcribe_audio(right_file, "Speaker 2", job_language)
            merged_plain, merged_ts = merge_transcripts((left_plain, left_ts), (right_plain, right_ts))
        elif mode == "diarization":
            sortformer_result = None
            if SORTFORMER_ENABLED:
                try:
                    sortformer_result = process_sortformer_diarization(
                        shared_jobs,
                        job_id,
                        job_dir,
                        job_input_file,
                        job_language,
                        model,
                    )
                except Exception as sort_exc:
                    logger.warning("Sortformer diarization workflow failed: %s", sort_exc, exc_info=True)
                    sortformer_result = None
            if sortformer_result is None:
                pipeline, pipeline_id = load_diarization_pipeline()

                def run_diarization(**kwargs):
                    result = pipeline(job_input_file, **kwargs)
                    annotation = getattr(result, "exclusive_speaker_diarization", None)
                    if annotation is None:
                        annotation = getattr(result, "speaker_diarization", result)
                    if not hasattr(annotation, "itertracks"):
                        raise AttributeError(
                            "Diarization output does not expose 'itertracks'; received "
                            f"{type(result)}"
                        )
                    return annotation

                diar_annotation = run_diarization()
                diar_segs = []
                for turn, _, speaker in diar_annotation.itertracks(yield_label=True):
                    diar_segs.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": str(speaker),
                    })
                diar_segs.sort(key=lambda x: x["start"])

                total_segments = len(diar_segs)
                current_info = dict(shared_jobs[job_id])
                current_info["progress_strategy"] = "segments"
                current_info["segments_total"] = total_segments
                current_info["segments_completed"] = 0
                current_info["diarization_model"] = pipeline_id
                current_info["speakers_detected"] = []
                if total_segments == 0:
                    current_info["progress"] = 100
                shared_jobs[job_id] = current_info

                segment_infos: list[dict] = []
                global_segments = []
                classifier = get_embedding_classifier()
                segments_completed = 0
                for i, seg in enumerate(diar_segs):
                    seg_start, seg_end, orig_spk = seg["start"], seg["end"], seg["speaker"]
                    spk_label = orig_spk
                    duration = max(seg_end - seg_start, 0.0)
                    if duration <= 0.05:
                        continue
                    segment_file = os.path.join(job_dir, f"segment_{i}.wav")
                    try:
                        run_ffmpeg_command([
                            "-ss",
                            f"{seg_start}",
                            "-t",
                            f"{duration}",
                            "-i",
                            job_input_file,
                            "-ac",
                            "1",
                            segment_file,
                        ])
                    except RuntimeError as exc:
                        logger.warning(
                            "FFmpeg failed for diarization segment %d: %s",
                            i,
                            exc,
                        )
                        continue
                    waveform_full = None
                    sample_rate = None
                    embedding = None
                    if classifier is not None:
                        try:
                            waveform_full, sample_rate = torchaudio.load(segment_file)
                            embedding = compute_waveform_embedding(waveform_full, sample_rate, classifier)
                        except Exception as emb_exc:
                            logger.debug("Embedding computation failed for %s: %s", segment_file, emb_exc)
                            waveform_full = None
                            sample_rate = None
                    seg_result = model.transcribe(
                        segment_file,
                        language=job_language,
                        fp16=USE_FP16_INFERENCE,
                    )
                    seg_segments = seg_result.get("segments", [])
                    if not seg_segments and seg_result.get("text"):
                        seg_segments = [{"start": 0.0, "end": duration, "text": seg_result.get("text", "")}]
                    if not seg_segments:
                        global_segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "speaker": spk_label,
                            "speaker_key": spk_label,
                            "text": seg_result.get("text", ""),
                            "segment_index": i,
                            "embedding": embedding,
                        })
                    else:
                        for s in seg_segments:
                            local_start = float(s.get("start", 0.0))
                            local_end = float(s.get("end", 0.0))
                            global_start = seg_start + local_start
                            global_end = seg_start + local_end
                            text = s.get("text", "")
                            sub_embedding = None
                            if (
                                classifier is not None
                                and waveform_full is not None
                                and sample_rate
                                and local_end > local_start
                            ):
                                start_sample = int(max(local_start, 0.0) * sample_rate)
                                end_sample = int(min(local_end, duration) * sample_rate)
                                end_sample = min(end_sample, waveform_full.size(-1))
                                if end_sample > start_sample:
                                    chunk_duration = (end_sample - start_sample) / max(sample_rate, 1)
                                    if chunk_duration >= EMBEDDING_MIN_DURATION_SECONDS:
                                        chunk_waveform = waveform_full[:, start_sample:end_sample]
                                        sub_embedding = compute_waveform_embedding(
                                            chunk_waveform,
                                            sample_rate,
                                            classifier,
                                        )
                            global_segments.append({
                                "start": global_start,
                                "end": global_end,
                                "speaker": spk_label,
                                "speaker_key": spk_label,
                                "text": text,
                                "segment_index": i,
                                "embedding": sub_embedding if sub_embedding is not None else embedding,
                            })
                    segment_infos.append({
                        "index": i,
                        "start": seg_start,
                        "end": seg_end,
                        "speaker_key": spk_label,
                        "speaker_name": spk_label,
                        "duration": duration,
                        "embedding": embedding,
                    })
                    segments_completed += 1
                    if total_segments > 0:
                        progress_fraction = min(segments_completed / total_segments, 1.0)
                        current_info = dict(shared_jobs[job_id])
                        current_info["segments_completed"] = segments_completed
                        current_info["progress"] = int(progress_fraction * 100)
                        shared_jobs[job_id] = current_info
                global_segments.sort(key=lambda x: x["start"])

                merged_plain, merged_ts, speakers_detected = finalize_diarization_segments(
                    global_segments,
                    segment_infos,
                    job_dir,
                )
                current_info = dict(shared_jobs[job_id])
                current_info["speakers_detected"] = speakers_detected
                shared_jobs[job_id] = current_info

            else:
                logger.info(
                    "Sortformer diarization produced %d segments across %d speakers.",
                    len(sortformer_result["global_segments"]),
                    len(sortformer_result["speakers_detected"]),
                )
                merged_plain = sortformer_result["merged_plain"]
                merged_ts = sortformer_result["merged_ts"]
                pipeline_id = sortformer_result["model_id"]
        else:
            raise ValueError("Invalid mode selected")

        out_plain = os.path.join(job_dir, "output_without_timestamps.txt")
        out_ts = os.path.join(job_dir, "output_with_timestamps.txt")
        with open(out_plain, "w", encoding="utf-8") as f:
            f.write("\n".join(merged_plain))
        with open(out_ts, "w", encoding="utf-8") as f:
            f.write("\n".join(merged_ts))

        final_info = dict(shared_jobs[job_id])
        final_info.update({
            "status": "done",
            "progress": 100,
            "eta": 0,
            "result_files": {"without": out_plain, "with": out_ts},
            "language": job_language
        })
        shared_jobs[job_id] = final_info
        save_job_metadata(job_id, final_info)
        actual_runtime = time.time() - start_time
        if not job_audio_duration:
            job_audio_duration = get_audio_duration(job_input_file)
        record_runtime_sample(mode, job_language, job_audio_duration, actual_runtime)
        logger.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}", exc_info=True)
        error_info = dict(shared_jobs[job_id])
        error_info.update({
            "status": "failed",
            "error": str(e),
            "language": job_language
        })
        shared_jobs[job_id] = error_info
        save_job_metadata(job_id, error_info)
    finally:
        updater_thread.join(timeout=2.0)

# ------------------------------------------------------
# Helper: Split stereo
# ------------------------------------------------------
def split_stereo(input_audio: str, output_dir: str):
    left_channel = os.path.join(output_dir, "left_speaker.wav")
    right_channel = os.path.join(output_dir, "right_speaker.wav")
    run_ffmpeg_command([
        "-i",
        input_audio,
        "-af",
        "pan=mono|c0=c0",
        left_channel,
    ])
    run_ffmpeg_command([
        "-i",
        input_audio,
        "-af",
        "pan=mono|c0=c1",
        right_channel,
    ])
    return left_channel, right_channel

# ------------------------------------------------------
# Helper: Transcribe a single file
# ------------------------------------------------------
def transcribe_audio(audio_file: str, speaker_label: str, language: str):
    model = whisper.load_model("large-v2")
    result = model.transcribe(
        audio_file,
        language=language,
        fp16=USE_FP16_INFERENCE,
    )
    segments = result.get("segments", [])
    plain_lines, ts_lines = [], []
    for seg in segments:
        start, end, text = seg["start"], seg["end"], seg["text"]
        plain_lines.append(f"{speaker_label}: {text}")
        ts_lines.append(f"[{start:.2f}s - {end:.2f}s] {speaker_label}: {text}")
    return plain_lines, ts_lines

# ------------------------------------------------------
# Helper: Merge transcripts from left/right
# ------------------------------------------------------
def merge_transcripts(left_data, right_data):
    def parse_ts(line: str) -> float:
        try:
            chunk = line.split("]")[0].lstrip("[")
            start_str = chunk.split("s -")[0]
            return float(start_str)
        except:
            return 0.0
    left_ts = left_data[1]
    right_ts = right_data[1]
    combined = []
    for l in left_ts:
        combined.append({"line": l, "ts": parse_ts(l)})
    for r in right_ts:
        combined.append({"line": r, "ts": parse_ts(r)})
    combined.sort(key=lambda x: x["ts"])
    merged_ts = [c["line"] for c in combined]
    merged_plain = []
    for entry in combined:
        parts = entry["line"].split("] ", 1)
        merged_plain.append(parts[1] if len(parts) == 2 else entry["line"])
    return merged_plain, merged_ts

# ------------------------------------------------------
# Index HTML Endpoint (with inline JS)
# ------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_content = r"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>Audio/Video Transcription</title>
  <script> tailwind = { config: { darkMode: 'class' } }; </script>
  <script src="https://cdn.tailwindcss.com"></script>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <script> function toggleDarkMode() { document.documentElement.classList.toggle("dark"); } </script>
</head>
<body class="bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100 min-h-screen">
  <div class="max-w-4xl mx-auto p-4">
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-3xl font-bold">Audio/Video Transcription</h1>
    </div>
    <form id="uploadForm" enctype="multipart/form-data" onsubmit="return false;" class="space-y-4 mb-8">
      <div>
        <label for="file" class="block mb-1 font-semibold">File:</label>
        <input name="file" id="file" type="file" accept="audio/*, video/*" required class="border rounded px-3 py-2 w-full focus:outline-none focus:ring dark:bg-gray-800 dark:border-gray-700" />
      </div>
      <div>
        <label for="mode" class="block mb-1 font-semibold">Select Mode:</label>
        <select name="mode" id="mode" class="border rounded px-3 py-2 w-full focus:outline-none focus:ring dark:bg-gray-800 dark:border-gray-700">
          <option value="one" selected>One Speaker Mode</option>
          <option value="stereo">Stereo Mode</option>
          <option value="diarization">Speaker Detection Mode</option>
        </select>
      </div>
      <div>
        <label for="language" class="block mb-1 font-semibold">Language:</label>
        <select name="language" id="language" class="border rounded px-3 py-2 w-full focus:outline-none focus:ring dark:bg-gray-800 dark:border-gray-700">
          <option value="en" selected>English</option>
          <option value="de">German</option>
        </select>
      </div>
      <div class="flex items-center space-x-4">
        <button type="button" id="uploadButton" onclick="submitForm()" class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition disabled:opacity-50">
          Upload
        </button>
        <div id="uploadProgress" class="hidden">
          <div class="w-64 bg-gray-200 dark:bg-gray-700 rounded">
            <div id="progressBar" class="h-2 bg-green-500 rounded" style="width: 0%"></div>
          </div>
          <span id="progressText" class="ml-2 font-medium">0%</span>
        </div>
      </div>
    </form>
    <hr class="my-6 border-gray-300 dark:border-gray-700" />
    <h2 class="text-xl font-semibold mb-2">Jobs</h2>
    <div id="totalTime" class="mb-4 font-semibold"></div>
    <div id="job-list" class="space-y-4"></div>
  </div>
  <script>
    const modeMapping = {
      "one": "One Speaker Mode",
      "stereo": "Stereo Mode",
      "diarization": "Speaker Detection Mode"
    };
    const languageMapping = {
      "en": "English",
      "de": "German"
    };
    function formatHMS(sec) {
      if (!sec || sec < 0) sec = 0;
      let h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60);
      return [h ? h + "h" : "", m ? m + "m" : "", s ? s + "s" : ""].filter(Boolean).join(" ") || "0s";
    }
    function parseJobs(jobs) {
      let arr = [];
      for (let jobId in jobs) { let job = jobs[jobId]; job["id"] = jobId; arr.push(job); }
      arr.sort((a, b) => (a.created_at || 0) - (b.created_at || 0));
      return arr;
    }
    function computeQueueEtas(jobsArr) {
      let cumulative = 0, totalQueueTime = 0;
      for (let j of jobsArr) {
        if (["done", "failed", "cancelled"].includes(j.status)) { j.etaFromNow = 0; continue; }
        if (j.status === "processing") { cumulative = j.eta || 0; j.etaFromNow = cumulative; }
        else { cumulative += (j.estimated_time || 0); j.etaFromNow = cumulative; }
        totalQueueTime = Math.max(totalQueueTime, j.etaFromNow);
      }
      return totalQueueTime;
    }
    function escapeHTML(value) {
      return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
    }
    function renderJobs(jobsDict) {
      const jobListDiv = document.getElementById("job-list");
      jobListDiv.innerHTML = "";
      const jobsArr = parseJobs(jobsDict);
      document.getElementById("totalTime").textContent = "Estimated time for entire queue: " + formatHMS(computeQueueEtas(jobsArr));
      jobsArr.forEach(job => {
        let card = document.createElement("div");
        card.className = "p-4 border border-gray-300 dark:border-gray-700 rounded shadow-sm";
        card.id = "job-" + job.id;
        let filename = job.original_filename || job.input_file.split(/(\\|\/)/g).pop();
        if (filename.startsWith('jobs/')) filename = filename.split('/').pop();
        let statusText = (job.status === "pending") ? "Queued" : job.status;
        let html = `<div class="font-semibold text-lg">${escapeHTML(filename)}</div>
                    <div class="text-sm text-gray-600 dark:text-gray-400">Mode: ${escapeHTML(modeMapping[job.mode] || job.mode)}</div>
                    <div class="text-sm text-gray-600 dark:text-gray-400">Language: ${escapeHTML(languageMapping[job.language] || job.language || 'English')}</div>
                    <div class="mt-2"><strong>Status:</strong> ${escapeHTML(statusText)}</div>`;
        if (job.status === "processing") {
          let pct = job.progress || 0;
          html += `<div class="mt-2 flex items-center space-x-2">
                     <div class="bg-gray-200 dark:bg-gray-700 w-48 h-2 rounded">
                       <div class="bg-blue-500 h-2 rounded" style="width: ${pct}%"></div>
                     </div>
                     <span class="text-sm">${pct}%</span>
                     <span class="text-sm"> (ETA: ${job.eta <= 1 && pct < 100 ? "finishing up..." : formatHMS(job.eta)})</span>
                   </div>`;
        }
        if (job.status === "pending") {
          let est = job.estimated_time ? formatHMS(job.estimated_time) : "unknown";
          html += `<div class="mt-2 text-sm">Estimated job time: ${est}</div>
                   <div class="mt-1 text-sm">Estimated finish in: ${formatHMS(job.etaFromNow)}</div>`;
        }
        if (job.status === "done") {
          html += `<div class="mt-2"><strong>Progress:</strong> 100%</div>
                   <div class="mt-1">
                     <a href="/download/${job.id}/without" class="text-blue-600 dark:text-blue-400 hover:underline">Download without timestamps</a>
                   </div>
                   <div>
                     <a href="/download/${job.id}/with" class="text-blue-600 dark:text-blue-400 hover:underline">Download with timestamps</a>
                   </div>
                   <div>
                     <button onclick="removeJob('${job.id}')" class="mt-2 px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700">Remove Job</button>
                   </div>`;
        }
        if (["failed", "cancelled"].includes(job.status)) {
          if (job.error) html += `<div class="mt-2 text-red-400"><strong>Error:</strong> ${escapeHTML(job.error)}</div>`;
          html += `<div>
                     <button onclick="removeJob('${job.id}')" class="mt-2 px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700">Remove Job</button>
                     <button onclick="restartJob('${job.id}')" class="mt-2 ml-2 px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">Restart Job</button>
                   </div>`;
        }
        if (job.status === "deleting") {
          html += `<div class="mt-2 text-sm text-gray-500">Deleting...</div>`;
        }
        card.innerHTML = html;
        jobListDiv.appendChild(card);
      });
    }
    // Optional bearer token, see the Security section of the README. The
    // browser asks for it the first time the server answers 401 and keeps it
    // for the tab's session.
    function authHeaders() {
      const token = sessionStorage.getItem("apiToken");
      return token ? { "Authorization": "Bearer " + token } : {};
    }
    function askForToken() {
      const token = prompt("This server requires an API token. Enter it to continue:");
      if (!token) return false;
      sessionStorage.setItem("apiToken", token.trim());
      return true;
    }
    async function fetchWithToken(url) {
      let response = await fetch(url, { headers: authHeaders() });
      if (response.status === 401 && askForToken()) {
        response = await fetch(url, { headers: authHeaders() });
      }
      return response;
    }
    async function removeJob(jobId) {
      if (!confirm('Are you sure you want to remove this job?')) return;
      try {
        const response = await fetchWithToken('/remove/' + jobId);
        if (!response.ok) {
          const error = await response.text();
          console.error('Failed to remove job:', error);
          alert('Failed to remove job: ' + error);
        }
      } catch (error) {
        console.error('Error removing job:', error);
        alert('Error removing job: ' + error.message);
      }
    }
    async function restartJob(jobId) {
      if (!confirm('Are you sure you want to restart this failed job?')) return;
      try {
        const response = await fetchWithToken('/restart/' + jobId);
        if (!response.ok) {
          const error = await response.text();
          console.error('Failed to restart job:', error);
          alert('Failed to restart job: ' + error);
        }
      } catch (error) {
        console.error('Error restarting job:', error);
        alert('Error restarting job: ' + error.message);
      }
    }
    function submitForm() {
      const form = document.getElementById("uploadForm");
      const formData = new FormData(form);
      const progressDiv = document.getElementById("uploadProgress");
      const progressBar = document.getElementById("progressBar");
      const progressText = document.getElementById("progressText");
      const uploadButton = document.getElementById("uploadButton");
      uploadButton.disabled = true;
      progressDiv.classList.remove("hidden");
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/upload", true);
      const token = sessionStorage.getItem("apiToken");
      if (token) xhr.setRequestHeader("Authorization", "Bearer " + token);
      xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
          const pct = (e.loaded / e.total) * 100;
          progressBar.style.width = pct + "%";
          progressText.textContent = Math.round(pct) + "%";
        }
      };
      xhr.onload = function() {
        if (xhr.status === 200) {
          form.reset();
          console.log("Upload successful:", xhr.response);
        } else if (xhr.status === 401 && askForToken()) {
          submitForm();
          return;
        } else {
          console.error("Upload failed:", xhr.responseText);
          alert("Upload failed: " + xhr.responseText);
        }
        progressDiv.classList.add("hidden");
        uploadButton.disabled = false;
        progressBar.style.width = "0%";
        progressText.textContent = "0%";
      };
      xhr.onerror = function() {
        console.error("Upload error");
        alert("Upload error");
        progressDiv.classList.add("hidden");
        uploadButton.disabled = false;
        progressBar.style.width = "0%";
        progressText.textContent = "0%";
      };
      xhr.send(formData);
    }
    function connectWebSocket() {
      const scheme = location.protocol === "https:" ? "wss://" : "ws://";
      const ws = new WebSocket(scheme + location.host + "/ws");
      ws.onmessage = function(event) {
        const jobs = JSON.parse(event.data);
        renderJobs(jobs);
      };
      ws.onclose = function() {
        console.log("WebSocket closed. Reconnecting...");
        setTimeout(connectWebSocket, 1000);
      };
      ws.onerror = function(err) {
        console.error("WebSocket error:", err);
      };
    }
    document.addEventListener("DOMContentLoaded", function() {
      fetch("/jobs")
        .then(resp => { if (!resp.ok) throw new Error("Failed to load jobs: " + resp.status); return resp.json(); })
        .then(data => renderJobs(data))
        .catch(err => console.error("Failed to load jobs:", err));
      connectWebSocket();
    });
  </script>
</body>
</html>
    """
    return html_content

# ------------------------------------------------------
# Upload Endpoint
# ------------------------------------------------------
@app.post("/upload", dependencies=[Depends(require_token)])
async def upload_file(
    file: UploadFile = File(...),
    mode: str = Form(...),
    language: str = Form("en")
):
    job_id = str(uuid.uuid4())
    selected_language = (language or "en").lower()
    display_name = safe_display_name(file.filename)
    extension = upload_extension(file.filename)
    logger.info(
        f"New upload request: job_id={job_id}, filename={display_name}, mode={mode}, language={selected_language}"
    )
    job_dir = os.path.join(JOB_BASE, job_id)
    file_location = os.path.join(job_dir, f"{uuid.uuid4()}{extension}")
    # Both path parts are generated here; the check keeps it that way if that ever changes.
    if Path(JOB_BASE).resolve() not in Path(file_location).resolve().parents:
        raise HTTPException(status_code=400, detail="Invalid upload path.")
    os.makedirs(job_dir, exist_ok=True)
    max_upload_bytes = MAX_UPLOAD_MB * BYTES_PER_MB
    received = 0
    try:
        with open(file_location, "wb") as f:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                received += len(chunk)
                if received > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the limit of {MAX_UPLOAD_MB} MB.",
                    )
                f.write(chunk)
    # BaseException so that a cancelled request also leaves no half-written job folder.
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    duration_sec = get_audio_duration(file_location)
    estimated_time = estimate_runtime_seconds(duration_sec, mode, selected_language)
    job_info = {
        "status": "pending",
        "input_file": file_location,
        "original_filename": display_name,
        "mode": mode,
        "language": selected_language,
        "progress": 0,
        "eta": None,
        "error": None,
        "audio_duration": duration_sec,
        "estimated_time": estimated_time,
        "created_at": time.time()
    }
    jobs[job_id] = job_info
    save_job_metadata(job_id, job_info)
    job_queue.put({
        "job_id": job_id,
        "file_location": file_location,
        "mode": mode,
        "language": selected_language
    })
    return {"job_id": job_id, "message": "File uploaded and job queued."}

# ------------------------------------------------------
# Restart Endpoint
# ------------------------------------------------------
@app.get("/restart/{job_id}", dependencies=[Depends(require_token)])
def restart_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] not in ("failed", "cancelled"):
        raise HTTPException(status_code=400, detail="Only failed or cancelled jobs can be restarted.")
    job["status"] = "pending"
    job["error"] = None
    job["progress"] = 0
    job["eta"] = None
    audio_duration = job.get("audio_duration") or get_audio_duration(job.get("input_file", ""))
    job["estimated_time"] = estimate_runtime_seconds(
        audio_duration,
        job.get("mode"),
        (job.get("language") or "en"),
    )
    save_job_metadata(job_id, job)
    job_queue.put({
        "job_id": job_id,
        "file_location": job["input_file"],
        "mode": job["mode"],
        "language": (job.get("language") or "en").lower()
    })
    return {"message": f"Job {job_id} restarted."}

# ------------------------------------------------------
# Remove Endpoint
# ------------------------------------------------------
@app.get("/remove/{job_id}", dependencies=[Depends(require_token)])
def remove_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    job["status"] = "deleting"
    save_job_metadata(job_id, job)
    def delete_job():
        job_dir = os.path.join(JOB_BASE, job_id)
        try:
            shutil.rmtree(job_dir)
        except Exception as e:
            logger.error(f"Error deleting job folder {job_id}: {e}")
        if job_id in job_processes:
            proc = job_processes[job_id]
            if proc.is_alive():
                proc.terminate()
            del job_processes[job_id]
        del jobs[job_id]
    threading.Thread(target=delete_job, daemon=True).start()
    return {"message": f"Job {job_id} is being deleted."}

# ------------------------------------------------------
# Download Endpoint
# ------------------------------------------------------
@app.get("/download/{job_id}/{file_type}")
def download_file(job_id: str, file_type: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    if "result_files" not in job or file_type not in job["result_files"]:
        raise HTTPException(status_code=400, detail="Invalid file type requested")
    file_path = job["result_files"][file_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    base_name = os.path.splitext(job.get("original_filename") or os.path.basename(job["input_file"]))[0]
    dl_name = f"{base_name}_transcription.txt" if file_type == "without" else f"{base_name}_transcription_timestamps.txt"
    return FileResponse(file_path, media_type="text/plain", filename=dl_name)

# ------------------------------------------------------
# Get Jobs Endpoint
# ------------------------------------------------------
@app.get("/jobs", response_class=JSONResponse)
def get_jobs():
    return dict(jobs)

# ------------------------------------------------------
# Queue Processing Function
# ------------------------------------------------------
def process_queue():
    while True:
        if shutdown_event.is_set():
            break
        try:
            if not job_queue.empty():
                with processing_lock:
                    job_data = job_queue.get()
                    if job_data:
                        process_job(
                            jobs,
                            job_data['job_id'],
                            job_data['file_location'],
                            job_data['mode'],
                            job_data.get('language', 'en'),
                            shutdown_event
                        )
            else:
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in queue processing: {str(e)}", exc_info=True)

# ------------------------------------------------------
# Startup: Initialize manager, load jobs, and start queue processing
# ------------------------------------------------------
def process_startup():
    global manager, jobs, shutdown_event
    multiprocessing.freeze_support()
    manager = multiprocessing.Manager()
    jobs = manager.dict()
    loaded_jobs = load_jobs_from_disk()
    for job_id, job_info in loaded_jobs.items():
        jobs[job_id] = job_info
    shutdown_event = multiprocessing.Event()
    threading.Thread(target=process_queue, daemon=True).start()

if __name__ == "__main__":
    logger.info("Starting application...")
    process_startup()
    logger.info(f"Listening on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
    logger.info("Shutting down manager...")
    if manager:
        manager.shutdown()
