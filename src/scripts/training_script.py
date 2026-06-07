"""
Script that prepares training data and runs the model training pipeline.

Basic workflow:
1. Load the latest training dataset from an S3 bucket.
2. Load and validate the uploaded CSV file received by the training API.
3. Merge both datasets and write them to the configured training data path.
4. Run the existing DesinfoVacinal training template.

The script sends status information to the training API through a TCP socket
connection when it is launched by api_training.py.
"""

import io
import os
import sys
import json
import socket
import logging

from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv

import boto3
import pandas as pd

from src.infra.schemas.model_config import parse_file
from src.infra.templates.desinfo_vacinal_template import DesinfoVacinalTemplate


# ----- Configuration -----
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

load_dotenv()

TRAINING_STATUS_HOST = os.getenv("TRAINING_STATUS_HOST", "localhost")
TRAINING_STATUS_PORT = int(os.getenv("TRAINING_STATUS_PORT", 9998))
TRAINING_DATA_S3_BUCKET = os.getenv("TRAINING_DATA_S3_BUCKET")
TRAINING_DATA_S3_PREFIX = os.getenv("TRAINING_DATA_S3_PREFIX", "")
TRAINING_DATA_S3_REGION = os.getenv("TRAINING_DATA_S3_REGION")
# -----


class StatusClient:
    def __init__(self, job_id: str | None):
        self.job_id = job_id
        self.socket: socket.socket | None = None

    def connect(self) -> None:
        if self.job_id is None:
            return

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((TRAINING_STATUS_HOST, TRAINING_STATUS_PORT))
        except Exception as exc:
            logging.warning("Could not connect to training API status socket: %s", exc)
            self.close()

    def send(self, status: str, stage: str, **extra) -> None:
        if self.socket is None:
            return

        payload = {
            "job_id": self.job_id,
            "status": status,
            "stage": stage,
            **extra,
        }

        try:
            self.socket.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except Exception as exc:
            logging.warning("Could not send training status update: %s", exc)
            self.close()

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None


def validate_training_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{source} must contain columns {sorted(required_columns)}. "
            f"Missing: {sorted(missing_columns)}. Found: {list(df.columns)}"
        )

    valid_df = df.dropna(subset=["text", "label"]).copy()
    if valid_df.empty:
        raise ValueError(f"{source} does not contain any valid rows with text and label values.")

    return valid_df


def get_latest_training_data() -> pd.DataFrame:
    """
    Download the latest training dataset CSV from S3 and return it as a DataFrame.

    Latest is defined by the newest LastModified timestamp among CSV objects under
    TRAINING_DATA_S3_BUCKET/TRAINING_DATA_S3_PREFIX.
    """
    if not TRAINING_DATA_S3_BUCKET:
        raise RuntimeError("TRAINING_DATA_S3_BUCKET environment variable is required.")

    client_kwargs = {}
    if TRAINING_DATA_S3_REGION:
        client_kwargs["region_name"] = TRAINING_DATA_S3_REGION

    s3_client = boto3.client("s3", **client_kwargs)
    paginator = s3_client.get_paginator("list_objects_v2")

    latest_object = None
    for page in paginator.paginate(
        Bucket=TRAINING_DATA_S3_BUCKET,
        Prefix=TRAINING_DATA_S3_PREFIX,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or not key.lower().endswith(".csv"):
                continue
            if latest_object is None or obj["LastModified"] > latest_object["LastModified"]:
                latest_object = obj

    if latest_object is None:
        prefix = TRAINING_DATA_S3_PREFIX or "<bucket root>"
        raise FileNotFoundError(
            f"No CSV training datasets found in s3://{TRAINING_DATA_S3_BUCKET}/{prefix}"
        )

    latest_key = latest_object["Key"]
    logging.info("Downloading latest training dataset from s3://%s/%s", TRAINING_DATA_S3_BUCKET, latest_key)

    response = s3_client.get_object(Bucket=TRAINING_DATA_S3_BUCKET, Key=latest_key)
    body = response["Body"].read()

    try:
        return pd.read_csv(io.BytesIO(body))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(body), encoding="latin-1")


def load_uploaded_training_data(uploaded_data_path: Path) -> pd.DataFrame:
    if not uploaded_data_path.exists():
        raise FileNotFoundError(f"Uploaded data file not found: {uploaded_data_path}")

    try:
        return pd.read_csv(uploaded_data_path)
    except UnicodeDecodeError:
        logging.warning("UTF-8 decode failed for uploaded data, trying latin-1 encoding")
        return pd.read_csv(uploaded_data_path, encoding="latin-1")


def merge_training_data(latest_df: pd.DataFrame, uploaded_df: pd.DataFrame) -> pd.DataFrame:
    latest_df = validate_training_dataframe(latest_df, "Latest S3 training dataset")
    uploaded_df = validate_training_dataframe(uploaded_df, "Uploaded training dataset")

    merged_df = pd.concat([latest_df, uploaded_df], ignore_index=True)
    merged_df = merged_df.dropna(subset=["text", "label"]).copy()
    merged_df["text"] = merged_df["text"].astype(str)
    merged_df = merged_df[merged_df["text"].str.strip() != ""]
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="last").reset_index(drop=True)

    if merged_df.empty:
        raise ValueError("Merged training dataset is empty after validation and deduplication.")

    return merged_df


def write_training_data(df: pd.DataFrame, data_dir: str) -> Path:
    output_path = Path(data_dir) / "train" / "data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logging.info("Merged training dataset written to %s with %d rows", output_path, len(df))
    return output_path


def parse_args():
    parser = ArgumentParser(description="Model Training Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file for training.",
    )
    parser.add_argument(
        "--uploaded-data",
        type=str,
        required=True,
        help="Path to the uploaded CSV file with new training rows.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=False,
        help="Training job identifier used by the API status endpoint.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_client = StatusClient(job_id=args.job_id)
    status_client.connect()

    try:
        status_client.send("in_progress", "loading_config")
        logging.info("Loading config file: %s", args.config)
        model_config = parse_file(args.config)

        status_client.send("in_progress", "loading_s3_dataset")
        latest_df = get_latest_training_data()
        logging.info("Latest S3 training dataset loaded with %d rows", len(latest_df))

        status_client.send("in_progress", "loading_uploaded_dataset")
        uploaded_df = load_uploaded_training_data(Path(args.uploaded_data))
        logging.info("Uploaded training dataset loaded with %d rows", len(uploaded_df))

        status_client.send("in_progress", "merging_dataset")
        merged_df = merge_training_data(latest_df, uploaded_df)
        output_path = write_training_data(merged_df, model_config.data.data_dir)

        status_client.send(
            "in_progress",
            "training",
            training_data_path=str(output_path),
            training_rows=len(merged_df),
        )
        logging.info("Starting training pipeline")
        template = DesinfoVacinalTemplate(config=model_config, inference=False)
        template.run()

        status_client.send("completed", "completed", training_rows=len(merged_df))
        logging.info("Training completed successfully")
        return 0
    except Exception as exc:
        logging.exception("Training failed")
        status_client.send("failed", "failed", error=str(exc))
        return 1
    finally:
        status_client.close()


if __name__ == "__main__":
    sys.exit(main())
