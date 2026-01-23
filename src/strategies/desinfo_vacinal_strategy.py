import logging
import shutil
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import tempfile
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_score, accuracy_score, 
    recall_score, f1_score, fbeta_score
)

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from src.infra.schemas.model_config import ModelConfig
from src.domain.datasets.desinfo_vacinal_dataset import DesinfoVacinalDataset
from src.domain.classifiers.desinfo_vacinal_model import DesinfoVacinalModel
from src.strategies.training_strategy import TrainingStrategy

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class DesinfoVacinalStrategy(TrainingStrategy):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.curr_seed = None
        self.X = []
        self.y = []
        self.labels = []
        self.predictions = []
        self.actual_labels = []
        self.losses = []
        self.best_fbeta = float('-inf')
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.pre_trained_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.output_path = Path("models") / Path(self.config.data.data_dir).name
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(1)
    
    def preprocess_data(self):
        data_dir = Path(self.config.data.data_dir)
        logging.info(f"Preprocessing data from directory: {data_dir}")
        
        try:
            df = pd.read_csv(data_dir / "train" / "data.csv")
        except FileNotFoundError:
            logging.error(f"Data file not found in {data_dir / 'train' / 'data.csv'}")
            raise RuntimeError("Data preprocessing failed due to missing file.")
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise RuntimeError("Data preprocessing failed due to an unexpected error.")
        
        self.X = [str(x) for x in df["text"]]
        
        self.labels = list(df["label"].value_counts().index)
        
        self.y = [self.labels.index(label) for label in df["label"]]
        
        logging.info(f"Data preprocessing completed. Number of samples: {len(self.X)}")
        logging.info(f"Labels found: {self.labels}")

    def load_dataset(self, seed: int):
        self.curr_seed = seed
        
        train_data, val_data, train_labels, val_labels = train_test_split(
            self.X, 
            self.y, 
            test_size=self.config.data.val_split, 
            random_state=seed
        )
        
        self.train_dataset = DesinfoVacinalDataset(
            texts=train_data,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.parameters.max_length
        )

        self.val_dataset = DesinfoVacinalDataset(
            texts=val_data,
            labels=val_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.parameters.max_length
        )
    
    def load_datamodel(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.parameters.batch_size,
            shuffle=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.parameters.batch_size,
            shuffle=False
        )

    def build_model(self):
        self.model = DesinfoVacinalModel(
            pretrained_model_name=self.config.pre_trained_model,
            num_labels=len(set(self.y))
        ).to(self.device)

    def train(self):
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.parameters.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=len(self.train_dataloader) * self.config.parameters.num_epochs
        )
        
        for epoch in range(self.config.parameters.num_epochs):
            self.losses = list()
            loss_fn = nn.CrossEntropyLoss()
            
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss = loss_fn(outputs, labels)
                self.losses.append(loss.item())
                
                loss.backward()
                
                optimizer.step()
                scheduler.step()

    def evaluate(self):
        self.model.eval()
    
        self.predictions = []
        self.actual_labels = []
        
        # Enters a context where gradients are not calculated
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Moving the tensors of the batch to the desired device (CPU or GPU)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Calling the forward method of the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Gets the predicted class by taking the argmax
                _, preds = torch.max(outputs, dim=1)
                
                # Move preds and labels to CPU and convert to lists
                self.predictions.extend(preds.cpu().tolist())
                self.actual_labels.extend(labels.cpu().tolist())
        
        fbeta = fbeta_score(self.actual_labels, self.predictions, beta=self.config.parameters.beta, average='weighted')
        
        if fbeta > self.best_fbeta:
            self.best_fbeta = fbeta
            self.save_best_model()
            self.save_metrics()

    def save_metrics(self):
        # Confusion matrix and metrics saving
        y_true = np.array(self.actual_labels)
        y_pred = np.array(self.predictions)
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.labels)))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        
        # Saving Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(8, 8))
        display.plot(ax=ax, values_format='d')
        plt.tight_layout()
        fig.savefig(self.output_path / "metrics" / "confusion_matrix.png", dpi=300)
        plt.close(fig)
        
        # Saving ROC Curve plot
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(self.output_path / "metrics" / "roc_curve.png", dpi=300)
        plt.close()
        
        # Saving Classification Report
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted')),
            "recall": float(recall_score(y_true, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted')),
            "fbeta_score": float(fbeta_score(y_true, y_pred, beta=self.config.parameters.beta, average='weighted')),
            "auc": float(auc)
        }
        
        with open(self.output_path / "metrics" / "classification_report.json", "w") as f:
            json.dump(metrics, f, indent=4)
    
    def save_best_model(self):
        """ Save the best model to disk atomically
        """
        def atomic_save(checkpoint, path):
            """ Avoids corrupted file in case of failure: writes tmp and does atomic rename
            """
            d = os.path.dirname(path)
            
            with tempfile.NamedTemporaryFile(delete=False, dir=d) as f:
                tmp = f.name
            
            torch.save(checkpoint, tmp)
            shutil.move(tmp, path)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "bert_model_name": self.config.pre_trained_model,
            "num_classes": len(self.labels),
            "tokenizer_name": self.config.pre_trained_model,
            "max_length": self.config.parameters.max_length,
            "seed": int(self.curr_seed),
            "fbeta": float(self.best_fbeta)
        }
        
        atomic_save(checkpoint, self.output_path / "best_model.pth")
    
    def deploy_best_model(self):
        """ Deploy the best model to MLflow
        """
        if not self.config.mlflow:
            logging.info("MLflow deployment is disabled in the configuration.")
            return
        
        model_name = "desinfo_vacinal_model"
        
        model_path = self.output_path / "best_model.pth"
        metrics_path = self.output_path / "metrics" / "classification_report.json"
        
        if not model_path.exists():
            logging.error(f"Best model file not found at {model_path}. Deployment aborted.")
            return
        if not metrics_path.exists():
            logging.error(f"Metrics file not found at {metrics_path}. Deployment aborted.")
            return
        
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)

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
            mlflow.log_param("seed", checkpoint.get("seed"))
            mlflow.log_param("lr", self.config.parameters.learning_rate)
            mlflow.log_param("batch_size", self.config.parameters.batch_size)
            mlflow.log_param("num_epochs", self.config.parameters.num_epochs)
            mlflow.log_param("beta", self.config.parameters.beta)
            mlflow.log_param("val_split", self.config.data.val_split)

            model = DesinfoVacinalModel(
                pretrained_model_name=checkpoint["bert_model_name"],
                num_labels=checkpoint["num_classes"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.cpu()

            # If you have Model Registry, you can also pass registered_model_name=...
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="desinfo_vacinal_model",
                registered_model_name=model_name
            )
        
        mlflow.log_metric("val_accuracy", metrics["accuracy"])
        mlflow.log_metric("val_precision", metrics["precision"])
        mlflow.log_metric("val_recall", metrics["recall"])
        mlflow.log_metric("val_f1", metrics["f1_score"])
        mlflow.log_metric("val_fbeta", metrics["fbeta_score"])
        mlflow.log_metric("val_auc", metrics["auc"])
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = DesinfoVacinalModel(
            pretrained_model_name=checkpoint["bert_model_name"],
            num_labels=checkpoint["num_classes"]
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        
        # Log the model to MLflow
        mlflow.pytorch.log_model(model, artifact_path="desinfo_vacinal_model")
        
        logging.info(f"Best model deployed to MLflow under experiment '{self.config.mlflow.experiment_name}'.")
        
        client = MlflowClient()
        latest = client.get_latest_versions(model_name)
        print("Latest versions:", [(v.version, v.current_stage) for v in latest])