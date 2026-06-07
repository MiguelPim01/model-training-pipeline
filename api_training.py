"""
API that trains the model through a RESTful interface.

Basic workflow:
1. Run uvicorn to start the API.
2. API starts and waits for a trigger to train the model.
3. When triggered, the API runs the training process in a child process.
4. The API monitors the training process and updates the database with the training status.

"""
import os
import sys
import json
import GPUtil
import shutil
import socket
import psutil
import uvicorn
import logging
import subprocess
import pandas as pd

from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from threading import Thread, Lock
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException

# ----- Configuration -----
load_dotenv()

CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "config/desinfo_vacinal.yaml")

API_HOST = os.getenv("API_HOST", "::")
API_PORT = int(os.getenv("API_PORT", 8000))
TRAINING_STATUS_HOST = os.getenv("TRAINING_STATUS_HOST", "localhost")
TRAINING_STATUS_PORT = int(os.getenv("TRAINING_STATUS_PORT", 9998))

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")) # Directory to store uploaded training data files
# -----

# ----- Basic checks -----
UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Ensure the upload directory exists
# -----

# ----- Global Variables -----
training_jobs: dict[str, dict[str, object]] = {}
active_job_id: str | None = None
jobs_lock = Lock()
# -----

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

def stream_child_logs(pipe, prefix: str) -> None:
    for line in iter(pipe.readline, ""):
        logging.info("%s %s", prefix, line.rstrip())
    pipe.close()

def update_job(job_id: str, **updates) -> None:
    with jobs_lock:
        if job_id not in training_jobs:
            training_jobs[job_id] = {}
        training_jobs[job_id].update(updates)
        training_jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

def validate_training_csv(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(status_code=400, detail="ERROR: Uploaded CSV is empty.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ERROR: Could not parse uploaded CSV: {exc}") from exc

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"ERROR: Uploaded CSV must contain columns {sorted(required_columns)}. Missing: {sorted(missing_columns)}.",
        )

    valid_rows = df.dropna(subset=["text", "label"])
    if valid_rows.empty:
        raise HTTPException(
            status_code=400,
            detail="ERROR: Uploaded CSV must contain at least one row with text and label values.",
        )

    return df

def training_child_process(data_path: Path, job_id: str) -> None:
    """
    Handler for child process to train the model when there is a trigger on an endpoint.
    """
    global active_job_id

    logging.info("Starting training child process for job %s", job_id)
    update_job(job_id, status="in_progress", stage="starting", return_code=None)

    server: socket.socket | None = None
    child: subprocess.Popen | None = None
    conn = None
    buffer = ""

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((TRAINING_STATUS_HOST, TRAINING_STATUS_PORT))
        server.listen(1)
        server.settimeout(1)

        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "src.scripts.training_script",
                "--config",
                CONFIG_FILE_PATH,
                "--uploaded-data",
                str(data_path),
                "--job-id",
                job_id,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "TRAINING_STATUS_HOST": TRAINING_STATUS_HOST,
                "TRAINING_STATUS_PORT": str(TRAINING_STATUS_PORT),
            },
        )

        if child.stdout is not None:
            Thread(
                target=stream_child_logs,
                args=(child.stdout, "[training-child]"),
                daemon=True,
            ).start()

        if child.stderr is not None:
            Thread(
                target=stream_child_logs,
                args=(child.stderr, "[training-child:stderr]"),
                daemon=True,
            ).start()

        while child.poll() is None:
            if conn is None:
                try:
                    conn, _addr = server.accept()
                    conn.settimeout(1)
                    logging.info("Training child process connected for status updates.")
                except TimeoutError:
                    continue
                except socket.timeout:
                    continue

            try:
                received_bytes = conn.recv(4096)
            except socket.timeout:
                continue

            if not received_bytes:
                continue

            buffer += received_bytes.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning("Invalid training status payload: %s", line)
                    continue

                status = payload.pop("status", None)
                if status:
                    payload["status"] = status
                update_job(job_id, **payload)

        return_code = child.wait()
        update_job(job_id, return_code=return_code)

        with jobs_lock:
            current_status = training_jobs.get(job_id, {}).get("status")

        if return_code == 0:
            if current_status != "completed":
                update_job(job_id, status="completed", stage="completed")
        else:
            update_job(job_id, status="failed", stage="failed", error=f"Training process exited with code {return_code}")
    except Exception as exc:
        logging.exception("Training job %s failed", job_id)
        update_job(job_id, status="failed", stage="failed", error=str(exc))
        if child is not None and child.poll() is None:
            child.terminate()
    finally:
        if conn is not None:
            conn.close()
        if server is not None:
            server.close()
        with jobs_lock:
            if active_job_id == job_id:
                active_job_id = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown logic.
    """
    # At startup
    
    yield
    
    # At shutdown

app = FastAPI(lifespan=lifespan)

# ----- Endpoints -----
@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint to verify that the API is running.
    """
    memory_info = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)

    gpus = GPUtil.getGPUs()
    gpu_info = [
        {
            "id": gpu.id,
            "name": gpu.name,
            "load": round(gpu.load * 100, 2),
            "memory_used": round(gpu.memoryUsed, 2),
            "memory_total": round(gpu.memoryTotal, 2),
            "memory_utilization": round(gpu.memoryUtil * 100, 2),
        }
        for gpu in gpus
    ]

    with jobs_lock:
        current_job = training_jobs.get(active_job_id) if active_job_id else None

    return {
        "healthy": True,
        "active_job_id": active_job_id,
        "active_job": current_job,
        "system_health": {
            "used_cpu_percent": cpu_usage,
            "used_memory_percent": memory_info.percent,
        },
        "gpus": gpu_info,
        "num_gpus": len(gpus),
    }

@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str) -> dict[str, object]:
    """
    Endpoint to get the status of a training job.
    
    Parameters:
        job_id: The unique identifier for the training job.
    
    Returns:
        json: Response containing the status of the training job. The keys are:
        - "job_id": The unique identifier for the training job.
        - "status": The current status of the training job, which can be "pending", "in_progress", "completed", or "failed".
    """
    with jobs_lock:
        job = training_jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")

    return {"job_id": job_id, **job}

@app.post("/train")
async def trigger_training(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Endpoint to trigger the training process.
    
    Parameters:
        file: A CSV file containing new training data.
    
    Returns:
        json: Response indicating that the training process has started, along with a unique job ID. The keys are:
        - "message": A string message confirming that the training process has started.
        - "job_id": A unique identifier for the training job, generated using uuid4.
    """
    # Error verification for the uploaded file
    if not file.filename:
        raise HTTPException(status_code=400, detail="ERROR: Missing filename.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="ERROR: Invalid file type. Only CSV files are allowed.")

    global active_job_id
    with jobs_lock:
        if active_job_id is not None:
            active_job = training_jobs.get(active_job_id, {})
            if active_job.get("status") in {"pending", "in_progress"}:
                raise HTTPException(
                    status_code=409,
                    detail=f"ERROR: Training job already running: {active_job_id}",
                )
            active_job_id = None

    # Download file
    job_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{job_id}.csv"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ERROR: Could not save uploaded file: {str(e)}")

    validate_training_csv(file_path)

    with jobs_lock:
        active_job_id = job_id
        training_jobs[job_id] = {
            "status": "pending",
            "stage": "queued",
            "uploaded_file": str(file_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "return_code": None,
            "error": None,
        }

    # Start training thread
    training_process = Thread(target=training_child_process, args=(file_path, job_id), daemon=True)
    training_process.start()

    return {
        "message": "Training process started.",
        "job_id": job_id
    }
# -----

def main():
    """
    Main function to run the API using Uvicorn.
    """
    uvicorn.run("api_training:app", host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
