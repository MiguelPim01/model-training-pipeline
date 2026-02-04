from pathlib import Path
import pandas as pd
import logging
import json
import sys

from mlflow.tracking import MlflowClient
import mlflow.pytorch
import mlflow.data
import mlflow

import torch

from src.infra.classifiers.desinfo_vacinal_model import DesinfoVacinalModel
from src.infra.schemas.model_config import ModelConfig

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

def deploy_run(config: ModelConfig, output_path: Path, device: str):
    """ Deploy the best model to MLflow
    """
    if not config.mlflow:
        logging.info("MLflow deployment is disabled in the configuration.")
        return
    
    model_name = "desinfo_vacinal_model"
    
    model_path = output_path / "best_model.pth"
    metrics_path = output_path / "metrics" / "classification_report.json"
    roc_curve_path = output_path / "metrics" / "roc_curve.png"
    confusion_matrix_path = output_path / "metrics" / "confusion_matrix.png"
    
    if not model_path.exists():
        logging.error(f"Best model file not found at {model_path}. Deployment aborted.")
        return
    if not metrics_path.exists():
        logging.error(f"Metrics file not found at {metrics_path}. Deployment aborted.")
        return
    
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    # Log models metrics
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    # One run for everything (metrics + model + artifacts)
    run_name = "deploy_best_model"
    with mlflow.start_run(run_name=run_name):
        # Log metrics safely
        metric_map = {
            "val_accuracy": "accuracy",
            "val_precision": "precision",
            "val_recall": "recall",
            "val_f1": "f1_score",
            "val_fbeta": "fbeta_score",
            "val_auc": "auc",
        }
        
        for mlflow_key, metrics_key in metric_map.items():
            if metrics_key in metrics:
                mlflow.log_metric(mlflow_key, float(metrics[metrics_key]))
            else:
                logging.warning(f"Metric '{metrics_key}' missing from {metrics_path}")

        # Log the metrics file itself as an artifact
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        
        # Log ROC curve if it exists
        if roc_curve_path.exists():
            mlflow.log_artifact(str(roc_curve_path), artifact_path="metrics")
            logging.info(f"ROC curve logged from {roc_curve_path}")
        else:
            logging.warning(f"ROC curve not found at {roc_curve_path}")

        # Log confusion matrix if it exists
        if confusion_matrix_path.exists():
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="metrics")
            logging.info(f"Confusion matrix logged from {confusion_matrix_path}")
        else:
            logging.warning(f"Confusion matrix not found at {confusion_matrix_path}")
            
        # Log dataset
        try:
            train_data_path = Path(config.data.data_dir) / "train" / "data.csv"
            if train_data_path.exists():
                df = pd.read_csv(train_data_path)
                
                dataset = mlflow.data.from_pandas(df, source=str(train_data_path))
                mlflow.log_input(dataset, context="training")
                
                logging.info(f"Training dataset logged from {train_data_path}")
            else:
                logging.warning(f"Training data not found at {train_data_path}")
        except Exception as e:
            logging.error(f"Failed to log dataset: {e}")

        checkpoint = torch.load(model_path, map_location="cpu")  # cpu is safer for portability

        # Log useful params
        mlflow.log_param("bert_model_name", checkpoint.get("bert_model_name"))
        mlflow.log_param("num_classes", checkpoint.get("num_classes"))
        mlflow.log_param("max_length", checkpoint.get("max_length"))
        mlflow.log_param("batch_size", config.parameters.batch_size)
        mlflow.log_param("num_epochs", config.parameters.num_epochs)
        mlflow.log_param("lr", config.parameters.learning_rate)
        mlflow.log_param("val_split", config.data.val_split)
        mlflow.log_param("seed", checkpoint.get("seed"))
        mlflow.log_param("beta", config.parameters.beta)
        
        model = DesinfoVacinalModel(
            pretrained_model_name=checkpoint["bert_model_name"],
            num_labels=checkpoint["num_classes"],
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.cpu()

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="desinfo_vacinal_model",
            registered_model_name=model_name
        )
    
    logging.info(f"Model deployed and registered under the name '{model_name}' in MLflow.")
    
    client = MlflowClient()
    latest = client.get_latest_versions(model_name)
    
    print("Latest versions:", [(v.version, v.current_stage) for v in latest])