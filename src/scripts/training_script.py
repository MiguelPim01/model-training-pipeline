"""
Script that prepares training data and runs the model training pipeline.

Basic workflow:
1. Load the latest training dataset from an S3 bucket.
2. Load and validate the uploaded CSV file received by the training API.
3. Merge both datasets and write them to the configured training data path.
4. Run the existing DesinfoVacinal training template.

The script sends status information to the training API by printing
newline-delimited JSON events to stdout when it is launched by api_training.py.
"""

import io
import os
import sys
import json
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

TRAINING_DATA_S3_BUCKET = os.getenv("TRAINING_DATA_S3_BUCKET")
TRAINING_DATA_S3_PREFIX = os.getenv("TRAINING_DATA_S3_PREFIX", "")
TRAINING_DATA_S3_REGION = os.getenv("TRAINING_DATA_S3_REGION")
ALLOW_FIRST_TRAINING_DATASET = os.getenv("ALLOW_FIRST_TRAINING_DATASET", "").lower() == "true"
STATUS_EVENT_PREFIX = "TRAINING_STATUS_JSON:"
# -----


def send_status_event(
    job_id: str | None,
    status: str,
    stage: str,
    message: str,
    **extra,
) -> None:
    payload = {
        "type": "training_status",
        "job_id": job_id,
        "status": status,
        "stage": stage,
        "message": message,
        **extra,
    }
    print(f"{STATUS_EVENT_PREFIX}{json.dumps(payload, default=str, separators=(',', ':'))}", flush=True)


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
        if ALLOW_FIRST_TRAINING_DATASET:
            logging.warning(
                "TRAINING_DATA_S3_BUCKET is not set. Continuing with uploaded data only."
            )
            return pd.DataFrame(columns=["text", "label"])
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
        message = f"No CSV training datasets found in s3://{TRAINING_DATA_S3_BUCKET}/{prefix}"
        if ALLOW_FIRST_TRAINING_DATASET:
            logging.warning("%s. Continuing with uploaded data only.", message)
            return pd.DataFrame(columns=["text", "label"])
        raise FileNotFoundError(message)

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
    uploaded_df = validate_training_dataframe(uploaded_df, "Uploaded training dataset")

    if latest_df.empty and ALLOW_FIRST_TRAINING_DATASET:
        logging.info("No previous S3 training dataset found; using uploaded data only.")
        latest_df = pd.DataFrame(columns=uploaded_df.columns)
    else:
        latest_df = validate_training_dataframe(latest_df, "Latest S3 training dataset")

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

    try:
        send_status_event(args.job_id, "in_progress", "loading_config", "Loading config file")
        logging.info("Loading config file: %s", args.config)
        model_config = parse_file(args.config)

        send_status_event(args.job_id, "in_progress", "loading_s3_dataset", "Loading latest training dataset from S3")
        latest_df = get_latest_training_data()
        if latest_df.empty and ALLOW_FIRST_TRAINING_DATASET:
            send_status_event(
                args.job_id,
                "in_progress",
                "loading_s3_dataset",
                "No previous S3 dataset found; using uploaded data only",
            )
        else:
            logging.info("Latest S3 training dataset loaded with %d rows", len(latest_df))

        send_status_event(args.job_id, "in_progress", "loading_uploaded_dataset", "Loading uploaded training dataset")
        uploaded_df = load_uploaded_training_data(Path(args.uploaded_data))
        logging.info("Uploaded training dataset loaded with %d rows", len(uploaded_df))

        send_status_event(args.job_id, "in_progress", "merging_dataset", "Merging training datasets")
        merged_df = merge_training_data(latest_df, uploaded_df)
        output_path = write_training_data(merged_df, model_config.data.data_dir)

        send_status_event(
            args.job_id,
            "in_progress",
            "training",
            "Starting training pipeline",
            training_data_path=str(output_path),
            training_rows=len(merged_df),
        )
        logging.info("Starting training pipeline")
        template = DesinfoVacinalTemplate(config=model_config, inference=False)
        template.run()

        model_path = Path("models") / Path(model_config.data.data_dir).name / "best_model.pth"
        if model_path.exists():
            send_status_event(
                args.job_id,
                "completed",
                "completed",
                "Training completed successfully",
                training_rows=len(merged_df),
                model_url=str(model_path),
            )
        else:
            logging.warning("Model artifact not found at %s", model_path)
            send_status_event(
                args.job_id,
                "completed",
                "completed",
                "Training completed successfully, but model artifact was not found",
                training_rows=len(merged_df),
            )
        logging.info("Training completed successfully")
        return 0
    except Exception as exc:
        logging.exception("Training failed")
        send_status_event(args.job_id, "failed", "failed", "Training failed", error=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
