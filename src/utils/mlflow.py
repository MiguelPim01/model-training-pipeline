from pathlib import Path
import logging
import json
import sys

from mlflow.tracking import MlflowClient
import mlflow.pytorch
import mlflow

import torch

from infra.classifiers.desinfo_vacinal_model import DesinfoVacinalModel
from infra.schemas.model_config import ModelConfig

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
            artifact_path="desinfo_vacinal_model",
            registered_model_name=model_name
        )
    
    # Log model and metrics outside the run for visibility
    mlflow.log_metric("val_precision", metrics["precision"])
    mlflow.log_metric("val_accuracy", metrics["accuracy"])
    mlflow.log_metric("val_fbeta", metrics["fbeta_score"])
    mlflow.log_metric("val_recall", metrics["recall"])
    mlflow.log_metric("val_f1", metrics["f1_score"])
    mlflow.log_metric("val_auc", metrics["auc"])
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = DesinfoVacinalModel(
        pretrained_model_name=checkpoint["bert_model_name"],
        num_labels=checkpoint["num_classes"]
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Log the model to MLflow
    mlflow.pytorch.log_model(model, artifact_path="desinfo_vacinal_model")
    
    logging.info(f"Best model deployed to MLflow under experiment '{config.mlflow.experiment_name}'.")
    
    client = MlflowClient()
    latest = client.get_latest_versions(model_name)
    
    print("Latest versions:", [(v.version, v.current_stage) for v in latest])