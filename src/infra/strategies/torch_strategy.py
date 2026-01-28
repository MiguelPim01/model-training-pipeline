from pathlib import Path
import pandas as pd
import numpy as np
import logging
import sys

from torch.utils.data import DataLoader
from torch import nn
import torch

from transformers import BertTokenizer, get_linear_schedule_with_warmup

from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.model_selection import train_test_split

from src.infra.datasets.desinfo_vacinal_dataset import DesinfoVacinalDataset
from src.infra.classifiers.desinfo_vacinal_model import DesinfoVacinalModel
from src.domain.strategies.training_strategy import ITrainingStrategy
from src.infra.schemas.model_config import ModelConfig

from src.utils.metrics import save_roc_curve, save_confusion_matrix, save_metrics_report
from src.utils.mlflow import deploy_run
from src.utils.model import save_best_model

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class TorchTrainingStrategy(ITrainingStrategy):
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
        data_dir.mkdir(parents=True, exist_ok=True)
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
            self.save_metrics()
            
            save_best_model(
                config=self.config,
                model=self.model,
                labels=self.labels,
                curr_seed=self.curr_seed,
                best_fbeta=self.best_fbeta,
                output_path=self.output_path
            )

    def save_metrics(self):
        output_metrics_dir = self.output_path / "metrics"
        output_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        y_true = np.array(self.actual_labels)
        y_pred = np.array(self.predictions)
        
        # Saving Confusion Matrix
        save_confusion_matrix(
            output_path=output_metrics_dir,
            y_true=y_true,
            y_pred=y_pred,
            labels=self.labels,
            display_labels=["True", "False"]
        )
        
        # Saving ROC Curve plot
        save_roc_curve(
            output_path=output_metrics_dir,
            y_true=y_true,
            y_pred=y_pred
        )
        
        # Saving Classification Report
        save_metrics_report(
            output_path=output_metrics_dir,
            y_true=y_true,
            y_pred=y_pred,
            beta=self.config.parameters.beta,
            auc=roc_auc_score(y_true, y_pred)
        )
    
    def deploy(self):
        """ Deploy the best model to MLflow
        """
        
        deploy_run(
            config=self.config,
            output_path=self.output_path,
            device=self.device
        )