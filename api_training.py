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
import shutil
import psutil
import uvicorn
import boto3
import logging
import subprocess
import pandas as pd
import psycopg

from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel
from threading import Thread, Lock
from urllib.parse import urlparse
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from src.infra.schemas.model_config import parse_file

try:
    import GPUtil
except ImportError:
    GPUtil = None

# ----- Configuration -----
load_dotenv()

CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "config/desinfo_vacinal.yaml")
DB_URL = os.getenv("DB_URL")
TRAINING_TABLE_NAME = os.getenv("TRAINING_TABLE_NAME")
TRAINING_STALE_TIMEOUT_MINUTES = int(os.getenv("TRAINING_STALE_TIMEOUT_MINUTES", 120))
TRAINING_HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("TRAINING_HEARTBEAT_INTERVAL_SECONDS", 60))
TRAINING_HEARTBEAT_LOG_INTERVAL_SECONDS = int(os.getenv("TRAINING_HEARTBEAT_LOG_INTERVAL_SECONDS", 0))

API_HOST = os.getenv("API_HOST", "::")
API_PORT = int(os.getenv("API_PORT", 8000))

DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "downloads/training"))
ALLOW_LOCAL_DATASET_PATHS = os.getenv("ALLOW_LOCAL_DATASET_PATHS", "").lower() == "true"
STATUS_EVENT_PREFIX = "TRAINING_STATUS_JSON:"
# -----

# ----- Basic checks -----
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
# -----

# ----- Global Variables -----
active_job_id: str | None = None
jobs_lock = Lock()
# -----

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)


class TrainingRequest(BaseModel):
    id: str | None = None
    by_user: str | None = None
    dataset_url: str | None = None
    version: str | None = None

def get_db_connection():
    if not DB_URL:
        raise RuntimeError("DB_URL environment variable is required for training status persistence.")

    return psycopg.connect(DB_URL, row_factory=dict_row)

def get_training_table_identifier() -> sql.Identifier:
    if not TRAINING_TABLE_NAME:
        raise RuntimeError(
            "TRAINING_TABLE_NAME environment variable is required because no training table "
            "name is defined in this repository."
        )

    return sql.Identifier(TRAINING_TABLE_NAME)

def fetch_training_job(job_id):
    table = get_training_table_identifier()

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {} WHERE {} = %s").format(
                    table,
                    sql.Identifier("id"),
                ),
                (job_id,),
            )
            return cursor.fetchone()

def append_log(existing_log, message) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"

    if not existing_log:
        return line

    return f"{existing_log}\n{line}"

def create_training_job(by_user, dataset_url, version=None):
    table = get_training_table_identifier()
    initial_log = append_log(None, f"Training job queued with dataset_url={dataset_url}")

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {}
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    RETURNING *
                    """
                ).format(
                    table,
                    sql.Identifier("by_user"),
                    sql.Identifier("dataset_url"),
                    sql.Identifier("status"),
                    sql.Identifier("log"),
                    sql.Identifier("version"),
                    sql.Identifier("createdAt"),
                    sql.Identifier("updatedAt"),
                ),
                (by_user, dataset_url, "pending", initial_log, version),
            )
            return cursor.fetchone()

def create_training_job_in_transaction(cursor, by_user, dataset_url, version=None):
    table = get_training_table_identifier()
    initial_log = append_log(None, f"Training job accepted with dataset_url={dataset_url}")

    cursor.execute(
        sql.SQL(
            """
            INSERT INTO {} (
                {},
                {},
                {},
                {},
                {},
                {},
                {}
            )
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            RETURNING *
            """
        ).format(
            table,
            sql.Identifier("by_user"),
            sql.Identifier("dataset_url"),
            sql.Identifier("status"),
            sql.Identifier("log"),
            sql.Identifier("version"),
            sql.Identifier("createdAt"),
            sql.Identifier("updatedAt"),
        ),
        (by_user, dataset_url, "pending", initial_log, version),
    )
    return cursor.fetchone()

def append_training_job_log_in_transaction(cursor, job_id, log_message):
    table = get_training_table_identifier()
    cursor.execute(
        sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
            sql.Identifier("log"),
            table,
            sql.Identifier("id"),
        ),
        (job_id,),
    )
    row = cursor.fetchone()
    existing_log = row["log"] if row else None
    cursor.execute(
        sql.SQL(
            """
            UPDATE {}
            SET {} = %s, {} = NOW()
            WHERE {} = %s
            RETURNING *
            """
        ).format(
            table,
            sql.Identifier("log"),
            sql.Identifier("updatedAt"),
            sql.Identifier("id"),
        ),
        (append_log(existing_log, log_message), job_id),
    )
    return cursor.fetchone()

def acquire_training_advisory_lock(cursor):
    cursor.execute(
        "SELECT pg_advisory_xact_lock(hashtext(%s))",
        ("model_training_global_lock",),
    )

def fail_stale_training_jobs(cursor):
    if TRAINING_STALE_TIMEOUT_MINUTES <= 0:
        return []

    table = get_training_table_identifier()
    stale_message = append_log(None, "Training job marked failed because it exceeded the stale timeout.")
    cursor.execute(
        sql.SQL(
            """
            UPDATE {}
            SET
                {} = %s,
                {} = COALESCE({}, '') || CASE WHEN {} IS NULL OR {} = '' THEN '' ELSE E'\n' END || %s,
                {} = NOW()
            WHERE {} IN (%s, %s)
              AND {} < NOW() - (%s * INTERVAL '1 minute')
            RETURNING {}, {}, {}
            """
        ).format(
            table,
            sql.Identifier("status"),
            sql.Identifier("log"),
            sql.Identifier("log"),
            sql.Identifier("log"),
            sql.Identifier("log"),
            sql.Identifier("updatedAt"),
            sql.Identifier("status"),
            sql.Identifier("updatedAt"),
            sql.Identifier("id"),
            sql.Identifier("status"),
            sql.Identifier("updatedAt"),
        ),
        ("failed", stale_message, "pending", "in_progress", TRAINING_STALE_TIMEOUT_MINUTES),
    )
    return cursor.fetchall() if hasattr(cursor, "fetchall") else []

def touch_training_job_updated_at(job_id, log_message=None):
    table = get_training_table_identifier()

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            if log_message is None:
                cursor.execute(
                    sql.SQL(
                        """
                        UPDATE {}
                        SET {} = NOW()
                        WHERE {} = %s
                        RETURNING *
                        """
                    ).format(
                        table,
                        sql.Identifier("updatedAt"),
                        sql.Identifier("id"),
                    ),
                    (job_id,),
                )
                return cursor.fetchone()

            cursor.execute(
                sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
                    sql.Identifier("log"),
                    table,
                    sql.Identifier("id"),
                ),
                (job_id,),
            )
            row = cursor.fetchone()
            existing_log = row["log"] if row else None
            cursor.execute(
                sql.SQL(
                    """
                    UPDATE {}
                    SET {} = %s, {} = NOW()
                    WHERE {} = %s
                    RETURNING *
                    """
                ).format(
                    table,
                    sql.Identifier("log"),
                    sql.Identifier("updatedAt"),
                    sql.Identifier("id"),
                ),
                (append_log(existing_log, log_message), job_id),
            )
            return cursor.fetchone()

def fetch_active_training_job_for_update(cursor):
    table = get_training_table_identifier()
    cursor.execute(
        sql.SQL(
            """
            SELECT {}, {}, {}
            FROM {}
            WHERE {} IN (%s, %s)
            ORDER BY {} ASC
            LIMIT 1
            FOR UPDATE
            """
        ).format(
            sql.Identifier("id"),
            sql.Identifier("status"),
            sql.Identifier("updatedAt"),
            table,
            sql.Identifier("status"),
            sql.Identifier("updatedAt"),
        ),
        ("pending", "in_progress"),
    )
    return cursor.fetchone()

def fetch_training_job_for_update(cursor, job_id):
    table = get_training_table_identifier()
    cursor.execute(
        sql.SQL("SELECT * FROM {} WHERE {} = %s FOR UPDATE").format(
            table,
            sql.Identifier("id"),
        ),
        (job_id,),
    )
    return cursor.fetchone()

def set_training_job_pending_in_transaction(cursor, job_id, log_message):
    table = get_training_table_identifier()
    cursor.execute(
        sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
            sql.Identifier("log"),
            table,
            sql.Identifier("id"),
        ),
        (job_id,),
    )
    row = cursor.fetchone()
    existing_log = row["log"] if row else None
    cursor.execute(
        sql.SQL(
            """
            UPDATE {}
            SET {} = %s, {} = %s, {} = NOW()
            WHERE {} = %s
            RETURNING *
            """
        ).format(
            table,
            sql.Identifier("status"),
            sql.Identifier("log"),
            sql.Identifier("updatedAt"),
            sql.Identifier("id"),
        ),
        ("pending", append_log(existing_log, log_message), job_id),
    )
    return cursor.fetchone()

def make_active_job_conflict(active_job):
    return HTTPException(
        status_code=409,
        detail={
            "detail": "A training job is already running.",
            "active_job": {
                "id": serialize_value(active_job.get("id")),
                "status": active_job.get("status"),
                "updatedAt": serialize_value(active_job.get("updatedAt")),
            },
        },
    )

def accept_training_job(request: TrainingRequest):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            acquire_training_advisory_lock(cursor)
            fail_stale_training_jobs(cursor)
            active_job = fetch_active_training_job_for_update(cursor)
            if active_job is not None:
                raise make_active_job_conflict(active_job)

            if request.id:
                job = fetch_training_job_for_update(cursor, request.id)
                if job is None:
                    raise HTTPException(status_code=404, detail=f"Training job not found: {request.id}")
                if not job.get("dataset_url"):
                    raise HTTPException(status_code=400, detail="ERROR: Training job must have dataset_url.")
                return set_training_job_pending_in_transaction(
                    cursor,
                    str(job["id"]),
                    "Training job accepted from existing database row.",
                )

            if not request.dataset_url:
                raise HTTPException(status_code=400, detail="ERROR: dataset_url is required when id is not provided.")

            return create_training_job_in_transaction(
                cursor,
                by_user=request.by_user,
                dataset_url=request.dataset_url,
                version=request.version,
            )

def serialize_value(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def serialize_training_job(job):
    if job is None:
        return None

    fields = [
        "id",
        "by_user",
        "dataset_url",
        "model_url",
        "status",
        "log",
        "version",
        "createdAt",
        "updatedAt",
    ]
    return {field: serialize_value(job.get(field)) for field in fields}

def update_training_job(job_id, status=None, log_message=None, model_url=None, extra=None):
    table = get_training_table_identifier()
    assignments = []
    values = []

    if status is not None:
        assignments.append(sql.SQL("{} = %s").format(sql.Identifier("status")))
        values.append(status)

    if model_url is not None:
        assignments.append(sql.SQL("{} = %s").format(sql.Identifier("model_url")))
        values.append(model_url)

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            if log_message is not None or extra:
                cursor.execute(
                    sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
                        sql.Identifier("log"),
                        table,
                        sql.Identifier("id"),
                    ),
                    (job_id,),
                )
                row = cursor.fetchone()
                existing_log = row["log"] if row else None

                if extra:
                    log_message = log_message or "Training status update"
                    log_message = f"{log_message} | extra={json.dumps(extra, default=str, sort_keys=True)}"

                assignments.append(sql.SQL("{} = %s").format(sql.Identifier("log")))
                values.append(append_log(existing_log, log_message))

            assignments.append(sql.SQL("{} = NOW()").format(sql.Identifier("updatedAt")))
            values.append(job_id)

            cursor.execute(
                sql.SQL("UPDATE {} SET {} WHERE {} = %s RETURNING *").format(
                    table,
                    sql.SQL(", ").join(assignments),
                    sql.Identifier("id"),
                ),
                values,
            )
            return cursor.fetchone()

def get_model_artifact_url() -> str:
    model_config = parse_file(CONFIG_FILE_PATH)
    return str(Path("models") / Path(model_config.data.data_dir).name / "best_model.pth")

def update_job(job_id: str, **updates) -> None:
    status = updates.pop("status", None)
    model_url = updates.pop("model_url", None)
    stage = updates.get("stage")
    error = updates.get("error")
    return_code = updates.get("return_code")

    log_parts = []
    if stage:
        log_parts.append(f"stage={stage}")
    if status:
        log_parts.append(f"status={status}")
    if return_code is not None:
        log_parts.append(f"return_code={return_code}")
    if error:
        log_parts.append(f"error={error}")

    log_message = "; ".join(log_parts) if log_parts else None
    update_training_job(job_id, status=status, log_message=log_message, model_url=model_url, extra=updates)

def handle_training_status_event(event: dict, expected_job_id: str) -> None:
    event_job_id = str(event.get("job_id")) if event.get("job_id") is not None else None
    if event_job_id != expected_job_id:
        logging.warning(
            "Ignoring training status event for unexpected job_id=%s; expected=%s",
            event_job_id,
            expected_job_id,
        )
        return

    status = event.get("status")
    stage = event.get("stage")
    message = event.get("message") or "Training status update"
    model_url = event.get("model_url")

    extra = {
        key: value
        for key, value in event.items()
        if key not in {"type", "job_id", "status", "message", "model_url"}
    }

    log_message = f"stage={stage}; status={status}; message={message}"
    update_training_job(
        expected_job_id,
        status=status,
        log_message=log_message,
        model_url=model_url,
        extra=extra,
    )

def stream_child_stdout(pipe, job_id: str) -> None:
    for line in iter(pipe.readline, ""):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if not stripped_line.startswith(STATUS_EVENT_PREFIX):
            logging.info("[training-child] %s", stripped_line)
            continue

        try:
            payload = json.loads(stripped_line[len(STATUS_EVENT_PREFIX):])
        except json.JSONDecodeError:
            logging.info("[training-child] %s", stripped_line)
            continue

        if payload.get("type") == "training_status":
            handle_training_status_event(payload, job_id)
        else:
            logging.info("[training-child] %s", stripped_line)

    pipe.close()

def stream_child_stderr(pipe, stderr_lines: list[str]) -> None:
    for line in iter(pipe.readline, ""):
        stripped_line = line.rstrip()
        stderr_lines.append(stripped_line)
        logging.info("[training-child:stderr] %s", stripped_line)
    pipe.close()

def wait_for_child_with_heartbeat(child: subprocess.Popen, job_id: str) -> int:
    heartbeat_count = 0
    interval = max(TRAINING_HEARTBEAT_INTERVAL_SECONDS, 1)
    log_interval = max(TRAINING_HEARTBEAT_LOG_INTERVAL_SECONDS, 0)

    while True:
        try:
            return child.wait(timeout=interval)
        except subprocess.TimeoutExpired:
            heartbeat_count += 1
            log_message = None
            if log_interval > 0 and heartbeat_count * interval >= log_interval:
                log_message = "Training heartbeat: child process is still running."
                heartbeat_count = 0

            try:
                touch_training_job_updated_at(job_id, log_message=log_message)
            except Exception:
                logging.exception("Could not update heartbeat for training job %s", job_id)

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

def get_download_path(job_id: str) -> Path:
    safe_job_id = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(job_id))
    if not safe_job_id:
        safe_job_id = str(uuid4())
    return DOWNLOAD_DIR / f"{safe_job_id}.csv"

def validate_resolved_dataset_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found after resolution: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Dataset path is not a file: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Dataset file must be a CSV: {file_path}")

    validate_training_csv(file_path)

def resolve_dataset_url_to_local_file(dataset_url: str, job_id: str) -> Path:
    destination = get_download_path(job_id)
    parsed_url = urlparse(dataset_url)

    update_training_job(
        job_id,
        log_message=f"Resolving dataset_url with scheme={parsed_url.scheme or 'local_path'}",
        extra={"stage": "resolving_dataset"},
    )

    try:
        if parsed_url.scheme in {"http", "https"}:
            raise ValueError("HTTP(S) dataset URLs are not enabled for this project.")
        elif parsed_url.scheme == "s3":
            bucket = parsed_url.netloc
            key = parsed_url.path.lstrip("/")
            if not bucket or not key:
                raise ValueError("S3 dataset_url must use the form s3://bucket/key")
            update_training_job(
                job_id,
                log_message=f"Downloading dataset from s3://{bucket}/{key}",
                extra={"stage": "resolving_dataset"},
            )
            boto3.client("s3").download_file(bucket, key, str(destination))
        else:
            if parsed_url.scheme and parsed_url.scheme != "file":
                raise ValueError(f"Unsupported dataset_url scheme: {parsed_url.scheme}")
            if not ALLOW_LOCAL_DATASET_PATHS:
                raise ValueError(
                    "Local dataset paths are disabled. Set ALLOW_LOCAL_DATASET_PATHS=true for development use."
                )
            source_path = Path(parsed_url.path if parsed_url.scheme == "file" else dataset_url)
            if not source_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {source_path}")
            shutil.copyfile(source_path, destination)
        validate_resolved_dataset_file(destination)
    except HTTPException:
        if destination.exists():
            destination.unlink()
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Could not resolve dataset_url to a local CSV: {exc}") from exc

    update_training_job(
        job_id,
        log_message=f"Dataset resolved to {destination}",
        extra={"stage": "dataset_resolved", "local_dataset_path": str(destination)},
    )
    return destination

def get_gpu_info() -> list[dict[str, object]]:
    if GPUtil is None:
        return []

    try:
        gpus = GPUtil.getGPUs()
    except Exception as exc:
        logging.warning("Could not read GPU info: %s", exc)
        return []

    return [
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

def training_child_process(dataset_url: str, job_id: str) -> None:
    """
    Handler for child process to train the model when there is a trigger on an endpoint.
    """
    global active_job_id

    logging.info("Starting training child process for job %s", job_id)
    update_job(job_id, status="in_progress", stage="resolving_dataset", return_code=None)

    child: subprocess.Popen | None = None
    stderr_lines: list[str] = []

    try:
        data_path = resolve_dataset_url_to_local_file(dataset_url, job_id)
        update_job(job_id, status="in_progress", stage="starting_training_script")

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
            env={**os.environ},
        )

        stdout_thread = None
        stderr_thread = None

        if child.stdout is not None:
            stdout_thread = Thread(
                target=stream_child_stdout,
                args=(child.stdout, job_id),
                daemon=True,
            )
            stdout_thread.start()

        if child.stderr is not None:
            stderr_thread = Thread(
                target=stream_child_stderr,
                args=(child.stderr, stderr_lines),
                daemon=True,
            )
            stderr_thread.start()

        return_code = wait_for_child_with_heartbeat(child, job_id)
        if stdout_thread is not None:
            stdout_thread.join(timeout=5)
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)

        update_job(job_id, return_code=return_code)

        current_job = fetch_training_job(job_id)
        current_status = current_job.get("status") if current_job else None

        if return_code == 0:
            model_url = get_model_artifact_url()
            if current_status != "completed":
                update_job(job_id, status="completed", stage="completed", model_url=model_url)
            else:
                update_training_job(
                    job_id,
                    model_url=model_url,
                    log_message=f"Model artifact available at {model_url}",
                )
        else:
            stderr_summary = "\n".join(stderr_lines[-20:])
            error_message = f"Training process exited with code {return_code}"
            if stderr_summary:
                error_message = f"{error_message}. stderr:\n{stderr_summary}"
            update_job(job_id, status="failed", stage="failed", error=error_message)
    except Exception as exc:
        logging.exception("Training job %s failed", job_id)
        update_job(job_id, status="failed", stage="failed", error=str(exc))
        if child is not None and child.poll() is None:
            child.terminate()
    finally:
        with jobs_lock:
            if active_job_id == job_id:
                active_job_id = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown logic.
    """
    # At startup
    if not DB_URL:
        raise RuntimeError("DB_URL environment variable is required for api_training.py.")
    if not TRAINING_TABLE_NAME:
        raise RuntimeError(
            "TRAINING_TABLE_NAME environment variable is required for api_training.py because "
            "the training table name is not defined in this repository."
        )
    
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
    gpu_info = get_gpu_info()

    current_job = fetch_training_job(active_job_id) if active_job_id else None

    return {
        "healthy": True,
        "active_job_id": active_job_id,
        "active_job": serialize_training_job(current_job),
        "system_health": {
            "used_cpu_percent": cpu_usage,
            "used_memory_percent": memory_info.percent,
        },
        "gpus": gpu_info,
        "num_gpus": len(gpu_info),
    }

@app.get("/train/status/{id}")
async def get_training_status(id: str) -> dict[str, object]:
    """
    Endpoint to get the status of a training job.
    
    Parameters:
        id: The unique identifier for the training job.
    
    Returns:
        json: Response containing the status of the training job. The keys are:
        - "id": The unique identifier for the training job.
        - "status": The current status of the training job, which can be "pending", "in_progress", "completed", or "failed".
    """
    try:
        job = fetch_training_job(id)
    except Exception as exc:
        logging.exception("Could not fetch training job %s", id)
        raise HTTPException(status_code=500, detail=f"ERROR: Could not fetch training job: {exc}") from exc

    if job is None:
        raise HTTPException(status_code=404, detail=f"Training job not found: {id}")

    return serialize_training_job(job)

@app.post("/train")
async def trigger_training(request: TrainingRequest) -> dict[str, str]:
    """
    Endpoint to trigger the training process.
    
    Returns a response containing the training row id and pending status.
    """
    global active_job_id
    try:
        job = accept_training_job(request)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logging.exception("Could not prepare training job")
        raise HTTPException(status_code=500, detail=f"ERROR: Could not prepare training job: {exc}") from exc

    job_id = str(job["id"])
    dataset_url = job["dataset_url"]

    with jobs_lock:
        active_job_id = job_id

    # Start training thread
    training_process = Thread(target=training_child_process, args=(dataset_url, job_id), daemon=True)
    training_process.start()

    return {
        "message": "Training process started.",
        "id": job_id,
        "status": "pending",
    }
# -----

def main():
    """
    Main function to run the API using Uvicorn.
    """
    uvicorn.run("api_training:app", host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
