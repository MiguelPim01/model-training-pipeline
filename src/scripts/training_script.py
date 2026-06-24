"""
Script that prepares training data and runs the model training pipeline.

Basic workflow:
1. Create socket connection with the training API.
2. Train the model.

The script sends status information to the database.
"""

import os
import sys
import logging

from dotenv import load_dotenv
from argparse import ArgumentParser

import boto3
import psycopg
import pandas as pd

from src.infra.schemas.model_config import parse_file
from src.infra.templates.desinfo_vacinal_template import DesinfoVacinalTemplate

# ----- Env variables -----
load_dotenv()

DB_URL = os.getenv("DB_URL")
TRAINING_TABLE_NAME = os.getenv("TRAINING_TABLE_NAME")
TRAINING_TABLE_SCHEMA = os.getenv("TRAINING_TABLE_SCHEMA") or "model"
CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "config/desinfo_vacinal.yaml")

# ----- Configuration -----
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)
# -----

def update_training_status(training_id: int, status: str, log: str):
    """
    Update the training status in the database.
    
    Args:
        training_id: The id of the training job in the database.
        status: The new status of the training job, which can be "pending", "in_progress", "completed", or "failed".
        log: A log message to be stored in the database.
    """
    
    # Create query to update the training status
    update_query = f"""
        UPDATE {TRAINING_TABLE_SCHEMA}."{TRAINING_TABLE_NAME}"
        SET status = %s, log = %s, "updatedAt" = NOW()
        WHERE id = %s;
    """
    values = (status, log, training_id)
    
    # Update the training status in the database
    try:
        with psycopg.connect(conninfo=DB_URL) as remote_session:
            with remote_session.cursor() as cursor:
                cursor.execute(query=update_query, params=values)
    except psycopg.OperationalError as e:
        logging.error(f"Error updating training status: {e}")

def parse_args():
    parser = ArgumentParser(description="Model Training Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file for training."
    )
    parser.add_argument(
        "--uploaded-data",
        type=str,
        required=True,
        help="Path to the uploaded CSV file with new training rows."
    )
    parser.add_argument(
        "--training-id",
        type=str,
        required=False,
        help="Training job identifier in the database."
    )

    return parser.parse_args()


def main() -> int:
    logging.info("Starting training script...")
    args = parse_args()
    
    logging.info(f"Loading config file: {args.config}")
    model_config = parse_file(args.config)

    try:
        # Starting and updating training status to in_progress
        logging.info("Starting training pipeline")
        update_training_status(
            training_id=int(args.training_id),
            status="in_progress",
            log="Training in progress."
        )
        
        # Starting training process
        template = DesinfoVacinalTemplate(config=model_config, inference=False)
        template.run()
        
        # Updating training status to completed
        update_training_status(
            training_id=int(args.training_id),
            status="completed",
            log="Training completed successfully."
        )
        logging.info("Training completed successfully")
    except Exception as exc:
        # Updating training status to failed
        update_training_status(
            training_id=int(args.training_id),
            status="failed",
            log=f"Training failed: {exc}"
        )
        
        logging.error(f"Training failed: {exc}")


if __name__ == "__main__":
    main()
