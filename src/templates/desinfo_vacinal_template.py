import logging
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, fbeta_score
)

from src.templates.training_pipeline_template import TrainingPipelineTemplate
from src.infra.schemas.model_config import ModelConfig
from src.domain.datasets.desinfo_vacinal_dataset import DesinfoVacinalDataset
from src.domain.classifiers.desinfo_vacinal_model import DesinfoVacinalModel

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class DesinfoVacinalTemplate(TrainingPipelineTemplate):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
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
        
        labels = list(df["label"].value_counts().index)
        
        self.y = [labels.index(label) for label in df["label"]]
        
        logging.info(f"Data preprocessing completed. Number of samples: {len(self.X)}")
        logging.info(f"Labels found: {labels} - {self.y}")

    def load_dataset(self):
        train_data, val_data, train_labels, val_labels = train_test_split(
            self.X, 
            self.y, 
            test_size=self.config.data.val_split, 
            random_state=42
        )
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.pre_trained_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

    def train_model(self):
        self.model = DesinfoVacinalModel(
            pretrained_model_name=self.config.pre_trained_model,
            num_labels=len(set(self.y))
        ).to(self.device)
        
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

    def evaluate_model(self):
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
    
        # Confusion matrix and metrics saving
        output_path = Path("models") / self.config.data.data_dir.parent.name / "metrics"
        output_path.mkdir(parents=True, exist_ok=True)
        
        y_true = np.array(self.actual_labels)
        y_pred = np.array(self.predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        # Saving Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(8, 8))
        display.plot(ax=ax, values_format='d')
        plt.tight_layout()
        fig.savefig(output_path / "confusion_matrix.png", dpi=300)
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
        plt.savefig(output_path / "roc_curve.png", dpi=300)
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
        
        with open(output_path / "classification_report.json", "w") as f:
            json.dump(metrics, f, indent=4)