import os
import logging
from datetime import datetime
import uuid
import shutil
import ffmpeg
import whisper
import time
import asyncio
import threading
import multiprocessing
from multiprocessing import Event
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from queue import Queue

# ------------------------------------------------------
# Logging Setup
# ------------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Global Setup
# ------------------------------------------------------
JOB_BASE = "jobs"
os.makedirs(JOB_BASE, exist_ok=True)

manager = None
jobs = None
job_processes = {}
shutdown_event = None
job_queue = Queue()
processing_lock = threading.Lock()

# ------------------------------------------------------
# Utility: Get audio duration via ffmpeg
# ------------------------------------------------------
def get_audio_duration(file_path: str) -> float:
    try:
        probe = ffmpeg.probe(file_path)
        duration_str = probe["format"]["duration"]
        return float(duration_str)
    except Exception as e:
        logger.error(f"Failed to get audio duration for {file_path}: {e}")
        return 0.0

# ------------------------------------------------------
# Main Job Processing Function
# ------------------------------------------------------
def process_job(shared_jobs, job_id: str, input_file_path: str, mode: str, shutdown_event: Event):
    """
    Pulls job from the shared dict, loads Whisper, processes either 'one', 'stereo', or 'diarization',
    updates progress, writes output files, sets status=done at the end.
    """
    logger.info(f"Starting job {job_id} in {mode} mode with file {input_file_path}")

    # 1) Mark job as "processing"
    job_info = dict(shared_jobs[job_id])
    estimated_transcription_time = job_info.get("estimated_time", 0.0)
    job_info["status"] = "processing"
    shared_jobs[job_id] = job_info

    start_time = time.time()

    # 2) A background thread that updates progress and ETA
    def background_progress_updater():
        while True:
            if shutdown_event.is_set():
                return
            current_info = dict(shared_jobs[job_id])
            # if job is done or cancelled, stop
            if current_info["status"] not in ("processing", "pending"):
                return

            elapsed = time.time() - start_time
            if estimated_transcription_time > 0:
                fraction = elapsed / estimated_transcription_time
            else:
                fraction = 0.0
            fraction = min(fraction, 1.0)

            progress_pct = int(fraction * 100)
            eta_seconds = max(int(estimated_transcription_time - elapsed), 0)

            current_info["progress"] = progress_pct
            current_info["eta"] = eta_seconds
            shared_jobs[job_id] = current_info

            if progress_pct >= 100:
                return
            time.sleep(1)

    updater_thread = threading.Thread(target=background_progress_updater, daemon=True)
    updater_thread.start()

    # 3) Prepare job folder, copy input, load Whisper model
    job_dir = os.path.join(JOB_BASE, job_id)
    job_input_file = os.path.join(job_dir, "input.wav")

    try:
        shutil.copy(input_file_path, job_input_file)
        model = whisper.load_model("large-v2")
        logger.info("Whisper model loaded in process_job")

        # 4) Process according to mode
        if mode == "one":
            # single speaker
            result = model.transcribe(job_input_file, language="en")
            segments = result.get("segments", [])
            merged_plain, merged_ts = [], []
            for seg in segments:
                start, end, text = seg["start"], seg["end"], seg["text"]
                merged_plain.append(text)
                merged_ts.append(f"[{start:.2f}s - {end:.2f}s] {text}")

        elif mode == "stereo":
            left_file, right_file = split_stereo(job_input_file, job_dir)
            left_plain, left_ts = transcribe_audio(left_file, "Speaker 1")
            right_plain, right_ts = transcribe_audio(right_file, "Speaker 2")
            merged_plain, merged_ts = merge_transcripts((left_plain, left_ts), (right_plain, right_ts))

        elif mode == "diarization":
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            diar = pipeline(job_input_file)

            diar_segs = []
            for turn, _, speaker in diar.itertracks(yield_label=True):
                diar_segs.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            diar_segs.sort(key=lambda x: x["start"])

            speaker_map = {}
            next_speaker = 1
            global_segments = []
            for i, seg in enumerate(diar_segs):
                seg_start, seg_end, orig_spk = seg["start"], seg["end"], seg["speaker"]
                if orig_spk not in speaker_map:
                    speaker_map[orig_spk] = f"Speaker {next_speaker}"
                    next_speaker += 1
                spk_label = speaker_map[orig_spk]

                segment_file = os.path.join(job_dir, f"segment_{i}.wav")
                duration = seg_end - seg_start
                ffmpeg.input(job_input_file, ss=seg_start, t=duration).output(
                    segment_file, ac=1
                ).run(overwrite_output=True, quiet=True)

                seg_result = model.transcribe(segment_file, language="en")
                seg_segments = seg_result.get("segments", [])
                for s in seg_segments:
                    global_start = seg_start + s["start"]
                    global_end = seg_start + s["end"]
                    text = s["text"]
                    global_segments.append({
                        "start": global_start,
                        "end": global_end,
                        "speaker": spk_label,
                        "text": text
                    })

            global_segments.sort(key=lambda x: x["start"])
            merged_plain, merged_ts = [], []
            for seg in global_segments:
                merged_plain.append(f"{seg['speaker']}: {seg['text']}")
                merged_ts.append(
                    f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}"
                )
        else:
            raise ValueError("Invalid mode selected")

        # 5) Write out final transcripts
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
            "result_files": {"without": out_plain, "with": out_ts}
        })
        shared_jobs[job_id] = final_info
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}", exc_info=True)
        error_info = dict(shared_jobs[job_id])
        if shutdown_event.is_set():
            error_info.update({
                "status": "cancelled",
                "error": "Job cancelled due to server shutdown."
            })
        else:
            error_info.update({
                "status": "failed",
                "error": str(e)
            })
        shared_jobs[job_id] = error_info

    finally:
        updater_thread.join(timeout=2.0)

# ------------------------------------------------------
# Helper: Split stereo
# ------------------------------------------------------
def split_stereo(input_audio: str, output_dir: str):
    left_channel = os.path.join(output_dir, "left_speaker.wav")
    right_channel = os.path.join(output_dir, "right_speaker.wav")
    ffmpeg.input(input_audio).output(left_channel, map="0:0", ac=1).run(
        overwrite_output=True, quiet=True
    )
    ffmpeg.input(input_audio).output(right_channel, map="0:1", ac=1).run(
        overwrite_output=True, quiet=True
    )
    return left_channel, right_channel

# ------------------------------------------------------
# Helper: Transcribe a single file
# ------------------------------------------------------
def transcribe_audio(audio_file: str, speaker_label: str):
    model = whisper.load_model("large-v2")
    result = model.transcribe(audio_file, language="en")
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
        if len(parts) == 2:
            merged_plain.append(parts[1])
        else:
            merged_plain.append(entry["line"])
    return merged_plain, merged_ts

# ------------------------------------------------------
# FastAPI lifespan, queue processing
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

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                            shutdown_event
                        )
            else:
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in queue processing: {str(e)}", exc_info=True)

# ------------------------------------------------------
# The HTML with minimal references to ID, plus "finishing up..." text
# Also with a mode -> friendly text mapping
# ------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_content = r"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>Audio/Video Transcription</title>
  <!-- Set darkMode:'class' before Tailwind -->
  <script>
    tailwind = {
      config: {
        darkMode: 'class'
      }
    };
  </script>
  <script src="https://cdn.tailwindcss.com"></script>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <script>
    function toggleDarkMode() {
      document.documentElement.classList.toggle("dark");
    }
  </script>
</head>
<body class="bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100 min-h-screen">
  <div class="max-w-4xl mx-auto p-4">
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-3xl font-bold">Audio/Video Transcription</h1>
    </div>

    <form id="uploadForm" enctype="multipart/form-data" onsubmit="return false;" class="space-y-4 mb-8">
      <div>
        <label for="file" class="block mb-1 font-semibold">File:</label>
        <input name="file" id="file" type="file" accept="audio/*, video/*" required
          class="border rounded px-3 py-2 w-full focus:outline-none focus:ring dark:bg-gray-800 dark:border-gray-700" />
      </div>

      <div>
        <label for="mode" class="block mb-1 font-semibold">Select Mode:</label>
        <select name="mode" id="mode"
          class="border rounded px-3 py-2 w-full focus:outline-none focus:ring dark:bg-gray-800 dark:border-gray-700">
          <option value="one" selected>One Speaker Mode</option>
          <option value="stereo">Stereo Mode</option>
          <option value="diarization">Speaker Detection Mode</option>
        </select>
      </div>

      <div class="flex items-center space-x-4">
        <button
          type="button"
          id="uploadButton"
          onclick="submitForm()"
          class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition disabled:opacity-50"
        >
          Upload
        </button>

        <!-- Upload progress bar -->
        <div id="uploadProgress" class="hidden">
          <div class="w-64 bg-gray-200 dark:bg-gray-700 rounded">
            <div
              id="progressBar"
              class="h-2 bg-green-500 rounded"
              style="width: 0%"
            ></div>
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
    // We'll map the internal 'mode' strings to user-friendly text:
    const modeMapping = {
      "one": "One Speaker Mode",
      "stereo": "Stereo Mode",
      "diarization": "Speaker Detection Mode"
    };

    // Format seconds => "Xh Ym Zs"
    function formatHMS(sec) {
      if (!sec || sec < 0) sec = 0;
      let h = Math.floor(sec / 3600);
      let m = Math.floor((sec % 3600) / 60);
      let s = Math.floor(sec % 60);
      let parts = [];
      if (h > 0) parts.push(h + "h");
      if (m > 0) parts.push(m + "m");
      if (s > 0) parts.push(s + "s");
      return parts.length > 0 ? parts.join(" ") : "0s";
    }

    // Convert jobs object => sorted array
    function parseJobs(jobs) {
      let arr = [];
      for (let jobId in jobs) {
        let job = jobs[jobId];
        job["id"] = jobId;
        arr.push(job);
      }
      arr.sort((a, b) => {
        if (!a.created_at) return 1;
        if (!b.created_at) return -1;
        return a.created_at - b.created_at;
      });
      return arr;
    }

    // For each job in sorted order:
    // If job is processing => leftover=job.eta
    // Else if queued => leftover += estimated_time
    function computeQueueEtas(jobsArr) {
      let cumulative = 0;
      let totalQueueTime = 0;
      for (let i = 0; i < jobsArr.length; i++) {
        let j = jobsArr[i];
        if (j.status === "done" || j.status === "failed" || j.status === "cancelled") {
          j.etaFromNow = 0;
          continue;
        }
        if (j.status === "processing") {
          cumulative = j.eta || 0;
          j.etaFromNow = cumulative;
        } else {
          cumulative += (j.estimated_time || 0);
          j.etaFromNow = cumulative;
        }
        if (j.etaFromNow > totalQueueTime) {
          totalQueueTime = j.etaFromNow;
        }
      }
      return totalQueueTime;
    }

    function renderJobs(jobsDict) {
      const jobListDiv = document.getElementById("job-list");
      jobListDiv.innerHTML = "";

      const jobsArr = parseJobs(jobsDict);
      const totalTime = computeQueueEtas(jobsArr);

      // Show total queue time
      const totalDiv = document.getElementById("totalTime");
      totalDiv.textContent = "Estimated time for entire queue: " + formatHMS(totalTime);

      for (let i = 0; i < jobsArr.length; i++) {
        let job = jobsArr[i];
        // Create a card
        let card = document.createElement("div");
        card.className = "p-4 border border-gray-300 dark:border-gray-700 rounded shadow-sm";

        // We only show the file name, not the job ID or path
        // Extract just the original filename without the job path
        let filename = job.input_file.split(/(\\|\/)/g).pop();
        if (filename.startsWith('jobs/')) {
            filename = filename.split('/').pop();
        }

        // Overwrite statusText if "pending" => "Queued"
        let statusText = job.status;
        if (job.status === "pending") statusText = "Queued";

        // Build up HTML
        let html = `<div class="font-semibold text-lg">${filename}</div>`;
        // Show the friendlier text for the mode:
        let displayMode = modeMapping[job.mode] || job.mode;
        html += `<div class="text-sm text-gray-600 dark:text-gray-400">Mode: ${displayMode}</div>`;
        html += `<div class="mt-2"><strong>Status:</strong> ${statusText}</div>`;

        if (job.status === "processing") {
          let pct = job.progress || 0;
          html += `
            <div class="mt-2 flex items-center space-x-2">
              <div class="bg-gray-200 dark:bg-gray-700 w-48 h-2 rounded">
                <div class="bg-blue-500 h-2 rounded" style="width: ${pct}%"></div>
              </div>
              <span class="text-sm">${pct}%</span>
          `;
          // Instead of showing 0s, show "finishing up..."
          if (job.eta !== null && job.eta !== undefined) {
            if (job.eta <= 1 && pct < 100) {
              html += ` <span class="text-sm">(ETA: finishing up...)</span>`;
            } else {
              html += ` <span class="text-sm">(ETA: ${formatHMS(job.eta)})</span>`;
            }
          }
          // Spinner
          html += `
              <div class="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
            </div>
          `;
        }
        if (statusText === "Queued") {
          let est = job.estimated_time ? formatHMS(job.estimated_time) : "unknown";
          html += `<div class="mt-2 text-sm">Estimated job time: ${est}</div>`;

          if (job.etaFromNow) {
            html += `<div class="mt-1 text-sm">Estimated finish in: ${formatHMS(job.etaFromNow)}</div>`;
          }
        }
        if (statusText === "done") {
          html += `
            <div class="mt-2"><strong>Progress:</strong> 100%</div>
            <div class="mt-1">
              <a href="/download/${job.id}/without" class="text-blue-600 dark:text-blue-400 hover:underline">Download without timestamps</a>
            </div>
            <div>
              <a href="/download/${job.id}/with" class="text-blue-600 dark:text-blue-400 hover:underline">Download with timestamps</a>
            </div>
            <div>
              <button onclick="removeJob('${job.id}')"
                class="mt-2 px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
              >
                Remove Job
              </button>
            </div>
          `;
        }
        if (statusText === "failed" || statusText === "cancelled") {
          if (job.error) {
            html += `<div class="mt-2 text-red-400"><strong>Error:</strong> ${job.error}</div>`;
          }
        }

        card.innerHTML = html;
        jobListDiv.appendChild(card);
      }
    }

    async function removeJob(jobId) {
      if (!confirm('Are you sure you want to remove this job?')) return;
      try {
        const response = await fetch('/remove/' + jobId);
        if (response.ok) {
          const result = await response.json();
          console.log(result.message);
        } else {
          const error = await response.text();
          console.error('Failed to remove job:', error);
          alert('Failed to remove job: ' + error);
        }
      } catch (error) {
        console.error('Error removing job:', error);
        alert('Error removing job: ' + error.message);
      }
    }

    async function submitForm() {
      const form = document.getElementById("uploadForm");
      const formData = new FormData(form);
      const progressDiv = document.getElementById("uploadProgress");
      const progressBar = document.getElementById("progressBar");
      const progressText = document.getElementById("progressText");
      const uploadButton = document.getElementById("uploadButton");

      try {
        uploadButton.disabled = true;
        progressDiv.classList.remove("hidden");

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/upload", true);

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
      } catch (error) {
        console.error("Upload error:", error);
        alert("Upload error: " + error.message);
        progressDiv.classList.add("hidden");
        uploadButton.disabled = false;
      }
    }

    function connectWebSocket() {
      const ws = new WebSocket("ws://" + location.host + "/ws");
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
      // load jobs
      fetch("/jobs")
        .then(resp => {
          if (!resp.ok) throw new Error("Failed to load jobs: " + resp.status);
          return resp.json();
        })
        .then(data => renderJobs(data))
        .catch(err => console.error("Failed to load jobs:", err));

      // start the WS
      connectWebSocket();
    });
  </script>
</body>
</html>
    """
    return html_content

# ------------------------------------------------------
# Endpoints
# ------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), mode: str = Form(...)):
    job_id = str(uuid.uuid4())
    logger.info(f"New upload request: job_id={job_id}, filename={file.filename}, mode={mode}")

    job_dir = os.path.join(JOB_BASE, job_id)
    os.makedirs(job_dir, exist_ok=True)

    file_location = os.path.join(job_dir, file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    duration_sec = get_audio_duration(file_location)
    speed_ratio = 1.0  # or adjust if your system is faster/slower than real-time
    estimated_time = duration_sec * speed_ratio

    jobs[job_id] = {
        "status": "pending",
        "input_file": file_location,
        "mode": mode,  # We'll map to user-friendly text in the UI
        "progress": 0,
        "eta": None,
        "error": None,
        "audio_duration": duration_sec,
        "estimated_time": estimated_time,
        "created_at": time.time()
    }

    job_queue.put({
        "job_id": job_id,
        "file_location": file_location,
        "mode": mode
    })

    return {"job_id": job_id, "message": "File uploaded and job queued."}

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

    base_name = os.path.splitext(os.path.basename(job["input_file"]))[0]
    if file_type == "without":
        dl_name = f"{base_name}_transcription.txt"
    else:
        dl_name = f"{base_name}_transcription_timestamps.txt"

    return FileResponse(file_path, media_type="text/plain", filename=dl_name)

@app.get("/remove/{job_id}")
def remove_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = os.path.join(JOB_BASE, job_id)
    try:
        shutil.rmtree(job_dir)
        if job_id in job_processes:
            proc = job_processes[job_id]
            if proc.is_alive():
                proc.terminate()
            del job_processes[job_id]
        del jobs[job_id]
        return {"message": f"Job {job_id} removed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_class=JSONResponse)
def get_jobs():
    return dict(jobs)

def process_startup():
    global manager, jobs, shutdown_event
    multiprocessing.freeze_support()
    manager = multiprocessing.Manager()
    jobs = manager.dict()
    shutdown_event = multiprocessing.Event()
    queue_thread = threading.Thread(target=process_queue, daemon=True)
    queue_thread.start()

if __name__ == "__main__":
    logger.info("Starting application...")
    process_startup()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Shutting down manager...")
    if manager:
        manager.shutdown()
