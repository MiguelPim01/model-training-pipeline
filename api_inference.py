"""
Inference API for making predictions to the database using the trained model.

Basic workflow:
1. Run uvicorn to start the API.
2. API runs the update thread and the inference process.
"""

import os
import sys
import json
import GPUtil
import socket
import psutil
import uvicorn
import logging
import psycopg
import subprocess

from pathlib import Path
from fastapi import FastAPI
from dotenv import load_dotenv
from threading import Thread, Event
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# ----- Configuration -----
load_dotenv()

DB_URL = os.getenv("DB_URL")
TIME_RANGE_DAYS = int(os.getenv("TIME_RANGE_DAYS", 365))
UPDATE_INTERVAL_MINUTES = int(os.getenv("UPDATE_INTERVAL_MINUTES", 120))

UPDATE_FILE_NAME = os.getenv("UPDATE_FILE_NAME", "update_records.json")
CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "config/desinfo_vacinal.yaml")

API_HOST = os.getenv("API_HOST", "::")
API_PORT = int(os.getenv("API_PORT", 8000))
# -----

# ----- Basic checks -----
if DB_URL is None: # Check if DB_URL is set
    logging.error("DB_URL environment variable is not set. Please set it in the .env file.")
    sys.exit(1)
else: # Test the connection
    try:
        with psycopg.connect(DB_URL) as remote_session:
            pass
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        sys.exit(1)
# -----


# ----- Global Variables -----
num_processed_posts = 0
num_unprocessed_posts = 0
healthy_classification = True # Indicates if the classification process by the model in child process is healthy
healthy_update = True # Indicates if the update process by the child process is healthy

stop_event = Event()
# -----

def update():
    """
    Computes statistics about the number of processed posts in the timestamp defined.
    
    Saves information in the file defined by UPDATE_FILE_NAME.
    """
    global num_unprocessed_posts
    global num_processed_posts
    
    update_file = Path(UPDATE_FILE_NAME)
    
    logging.info("Starting the update thread...")
    
    query_processed = f"""
        SELECT count(*)
        FROM "Post" p
        WHERE p."predicted_pa" IS NOT NULL AND p."time" >= CURRENT_DATE - INTERVAL '1 day' * {TIME_RANGE_DAYS};
    """
    
    query_not_processed = f"""
        SELECT count(*)
        FROM "Post" p
        WHERE p."predicted_pa" IS NULL AND p."time" >= CURRENT_DATE - INTERVAL '1 day' * {TIME_RANGE_DAYS};
    """
    
    while not stop_event.is_set():
        
        with psycopg.connect(conninfo=DB_URL) as remote_session:
            with remote_session.cursor() as cursor:
                try:
                    cursor.execute(query_processed)
                    num_processed_posts = cursor.fetchone()[0]
                    
                    cursor.execute(query_not_processed)
                    num_unprocessed_posts = cursor.fetchone()[0]
                except psycopg.OperationalError as e:
                    print(f"Operational error while fetching post counts: {e}")
                except Exception as e:
                    print(f"Error while fetching post counts: {e}")
        
        info = {
            "time": datetime.now(timezone.utc).isoformat(),
            "processed_posts": num_processed_posts,
            "unprocessed_posts": num_unprocessed_posts
        }
        
        if not update_file.exists():
            curr_history = list()
        else:
            with open(update_file, "r") as f:
                try:
                    curr_history = json.load(f)
                except json.JSONDecodeError as e: # File is empty or contains invalid JSON
                    logging.warning(f"Error decoding JSON from {update_file}: {e}")
                    
                    curr_history = list()
        
        curr_history.append(info)
        
        # Write back (overwrite) the file
        with open(update_file, "w") as f:
            json.dump(curr_history, f, indent=2)
        
        # Wait or exit early if stop_event is set
        if stop_event.wait(timeout=UPDATE_INTERVAL_MINUTES * 60):
            break

def inference_child_process_handler():
    """
    Handler for the child process that performs inference and updates the database.
    """
    logging.info("Starting the inference child process...")
    
    global healthy_classification
    global healthy_update
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("localhost", 9999))
    server.listen(1)
    
    # sys.executable uses the exact same python that the current script is running on
    child = subprocess.Popen([sys.executable, "-m", "src.scripts.inference_script", "--config", CONFIG_FILE_PATH])
    conn, addr = server.accept() # wait for connection from child process
    
    logging.info("Child process started and connected.")
    
    # While child process is running
    while child.poll() is None:
        received_bytes = conn.recv(1024)
        if not received_bytes:
            continue

        data = json.loads(received_bytes.decode())
        healthy_classification = data.get("healthy_classification", healthy_classification)
        healthy_update = data.get("healthy_update", healthy_update)

    logging.info("Child process handler thread finished.")
    
    conn.close()
    server.close()
    
    healthy_classification = False
    healthy_update = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown logic.
    """
    # At startup
    update_thread = Thread(target=update)
    update_thread.start()
    
    inference_child_process_thread = Thread(target=inference_child_process_handler)
    inference_child_process_thread.start()
    
    yield
    
    # At shutdown
    stop_event.set()
    update_thread.join()
    inference_child_process_thread.join()

app = FastAPI(lifespan=lifespan)

# ----- Endpoints -----
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    # Get CPU and Memory info using psutil
    memory_info = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)

    # Get GPU info using GPUtil
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "id": gpu.id,
            "name": gpu.name,
            "load": round(gpu.load * 100, 2),  # Convert to percentage
            "memory_used": round(gpu.memoryUsed, 2),
            "memory_total": round(gpu.memoryTotal, 2),
            "memory_utilization": round(gpu.memoryUtil * 100, 2)  # Convert to percentage
        })

    return {
        "healthy_classification": healthy_classification,
        "healthy_update": healthy_update,
        "system_health": {
            "used_cpu_percent": cpu_usage,
            "used_memory_percent": memory_info.percent
        },
        "gpus": gpu_info,
        "num_gpus": len(gpus),
        "processed_posts": num_processed_posts,
        "unprocessed_posts": num_unprocessed_posts
    }
# -----

def main():
    """
    Main function to run the API using Uvicorn.
    """
    uvicorn.run("api_inference:app", host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()