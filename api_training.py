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
import boto3
import psutil
import GPUtil
import psycopg
import uvicorn
import logging
import subprocess
import pandas as pd

from pathlib import Path
from threading import Thread
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# ----- Configuration -----
load_dotenv()

DB_URL = os.getenv("DB_URL")
TRAINING_TABLE_NAME = os.getenv("TRAINING_TABLE_NAME", "Table")
TRAINING_TABLE_SCHEMA = os.getenv("TRAINING_TABLE_SCHEMA") or "model"
CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "config/desinfo_vacinal.yaml")

API_HOST = os.getenv("API_HOST", "::")
API_PORT = int(os.getenv("API_PORT", 8000))

ORIGIN_DATA_PATH = "data/desinfo_vacinal/train/data.csv"
UPLOADED_TRAIN_DATA_PATH = "data/uploaded/data.csv"
# -----

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class TrainRequest(BaseModel):
    by_user: str
    dataset_s3_location: str

def get_gpu_info() -> list[dict[str, object]]:
    if GPUtil is None:
        return []

    try:
        gpus = GPUtil.getGPUs()
    except Exception as exc:
        logging.warning("Could not read GPU info: %s", exc)
        return []

    return [{
            "id": gpu.id,
            "name": gpu.name,
            "load": round(gpu.load * 100, 2),
            "memory_used": round(gpu.memoryUsed, 2),
            "memory_total": round(gpu.memoryTotal, 2),
            "memory_utilization": round(gpu.memoryUtil * 100, 2),
        } for gpu in gpus
    ]

def load_s3_dataset(dataset_url: str) -> pd.DataFrame:
    """
    Loads dataset from s3 bucket.

    Args:
        dataset_url (str): Link to s3 bucket

    Returns:
        pd.DataFrame: Dataframe to be used in training.
    """
    s3_path = dataset_url.replace("s3://", "", 1)
    bucket_name, object_key = s3_path.split("/", 1)

    logging.info(f"Loading dataset from S3 bucket={bucket_name}, key={object_key}")

    s3_client = boto3.client("s3")
    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=object_key,
    )

    dataframe = pd.read_csv(response["Body"])

    logging.info(f"Dataset loaded successfully. Shape: {dataframe.shape}")

    return dataframe

def create_full_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the original training dataset, appends the new S3 dataset,
    and returns the full dataset for training.

    Args:
        data (pd.DataFrame): New training data loaded from S3.

    Returns:
        pd.DataFrame: Original dataset + new S3 dataset.
    """
    logging.info(f"Loading original dataset from {ORIGIN_DATA_PATH}")

    original_data = pd.read_csv(ORIGIN_DATA_PATH)

    if set(original_data.columns) != set(data.columns):
        raise ValueError(
            "S3 dataset columns do not match original dataset columns. "
            f"Original columns: {list(original_data.columns)}. "
            f"S3 columns: {list(data.columns)}."
        )

    data = data[original_data.columns]

    full_dataset = pd.concat(
        [original_data, data],
        ignore_index=True,
    )

    logging.info(
        f"Full dataset created. "
        f"Original shape: {original_data.shape}. "
        f"New data shape: {data.shape}. "
        f"Full shape: {full_dataset.shape}."
    )

    return full_dataset

def training_child_process(dataset_url: str, training_id: str) -> None:
    """
    Handler for child process to train the model when there is a trigger on an endpoint.
    """
    logging.info(f"Starting training process for dataset_url: {dataset_url} and training_id: {training_id}")
    
    try:
        # Loads data from the S3 Bucket
        data = load_s3_dataset(dataset_url=dataset_url)
        
        # Creates full dataset
        full_dataset = create_full_dataset(data=data)
        
        # Stores the full dataset into memory
        data_path = Path(UPLOADED_TRAIN_DATA_PATH)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_dataset.to_csv(data_path, index=False)
        
        # Command to run the training process
        command = [
            sys.executable,
            "-m",
            "src.scripts.training_script",
            "--config",
            CONFIG_FILE_PATH,
            "--uploaded-data",
            str(data_path),
            "--training-id",
            str(training_id),
        ]
        
        logging.info(f"Running training command: {' '.join(command)}")

        subprocess.run(
            command,
            check=True,
        )
        
        logging.info(f"Training process {training_id} finished successfully")
    except subprocess.CalledProcessError as e:
        logging.exception(
            f"Training script failed for training_id={training_id}. "
            f"Return code: {e.returncode}"
        )

    except Exception as e:
        logging.exception(
            f"Error running training process {training_id}: {e}"
        )

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
    gpu_info = get_gpu_info()

    return {
        "healthy": True,
        "system_health": {
            "used_cpu_percent": cpu_usage,
            "used_memory_percent": memory_info.percent,
        },
        "gpus": gpu_info,
        "num_gpus": len(gpu_info),
    }

@app.post("/train")
async def trigger_training(request: TrainRequest) -> dict[str, str]:
    """
    Endpoint to trigger the training process.
    
    Returns a response containing the training row id and pending status.
    """
    # Creating query to insert the new training process into the database
    insert_query = f"""
        INSERT INTO {TRAINING_TABLE_SCHEMA}."{TRAINING_TABLE_NAME}"
            (by_user, dataset_url, model_url, status, log, "createdAt", "updatedAt")
        VALUES
            (%s, %s, %s, %s, %s, NOW(), NOW())
        RETURNING id;
    """
    values = (request.by_user, request.dataset_url, None, "pending", "Training process created.")
    
    logging.info(f"Query to be executed: {insert_query % values}")
    
    # Creating training row in the database and getting its id
    try:
        with psycopg.connect(conninfo=DB_URL) as remote_session:
            with remote_session.cursor() as cursor:
                cursor.execute(query=insert_query, params=values)

                training_id = cursor.fetchone()[0]
    except psycopg.OperationalError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database operational error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating training process: {e}",
        )

    # Creating Thread to run the training process
    training_process = Thread(
        target=training_child_process,
        args=(
            request.dataset_s3_location,
            training_id,
        ),
        daemon=True,
    )

    training_process.start()

    return {
        "training_id": training_id,
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
