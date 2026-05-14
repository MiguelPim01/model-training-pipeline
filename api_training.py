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
import psycopg
import subprocess

from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from threading import Thread, Event
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException

# ----- Configuration -----
load_dotenv()

DB_URL = os.getenv("DB_URL")
CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "config/desinfo_vacinal.yaml")

API_HOST = os.getenv("API_HOST", "::")
API_PORT = int(os.getenv("API_PORT", 8000))

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")) # Directory to store uploaded training data files
# -----

# ----- Basic checks -----
UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Ensure the upload directory exists
# -----

# ----- Global Variables -----
training_status = None
# -----

def training_child_process(data_path: Path):
    """
    Handler for child process to train the model when there is a trigger on an endpoint.
    """
    # TODO: Implement socket communication to check on the trianing process
    pass

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
@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str) -> dict[str, str]:
    """
    Endpoint to get the status of a training job.
    
    Parameters:
        job_id: The unique identifier for the training job.
    
    Returns:
        json: Response containing the status of the training job. The keys are:
        - "job_id": The unique identifier for the training job.
        - "status": The current status of the training job, which can be "pending", "in_progress", "completed", or "failed".
    """
    # TODO: Implement real logic
    return {
        "job_id": job_id,
        "status": "pending"
    }

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
    
    # Download file
    job_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{job_id}.csv"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ERROR: Could not save uploaded file: {str(e)}")

    # Start training thread
    training_process = Thread(target=training_child_process, args=(file_path,), daemon=True)
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