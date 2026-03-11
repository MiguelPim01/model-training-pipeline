from pathlib import Path
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
from src.domain.pipelines.data_pipeline import ProcessedData

from src.utils.metrics import save_roc_curve, save_confusion_matrix, save_metrics_report, save_training_history
from src.utils.model import save_best_model
from src.utils.mlflow import deploy_run

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    stream=sys.stdout,
)


class TorchTrainingStrategy(ITrainingStrategy):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.curr_seed = None
        self.num_labels = None
        self.labels = []
        self.predictions = []
        self.probabilities = []
        self.actual_labels = []
        self.losses = []
        self.history = []
        self.running_loss = 0.0
        self.val_loss = 0.0
        self.best_fbeta = float("-inf")

        self.tokenizer = BertTokenizer.from_pretrained(self.config.pre_trained_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_path = Path("models") / Path(self.config.data.data_dir).name
        self.output_path.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(1)

    def load_dataset(self, data: ProcessedData, seed: int):
        self.curr_seed = seed
        self.num_labels = len(set(data.labels))
        self.labels = data.labels

        train_data, val_data, train_labels, val_labels = train_test_split(
            data.X, data.y, test_size=self.config.data.val_split, random_state=seed
        )

        self.train_dataset = DesinfoVacinalDataset(
            texts=train_data,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.parameters.max_length,
        )

        self.val_dataset = DesinfoVacinalDataset(
            texts=val_data,
            labels=val_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.parameters.max_length,
        )

    def load_datamodel(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.parameters.batch_size,
            shuffle=True,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.parameters.batch_size,
            shuffle=False,
        )

    def build_model(self):
        self.model = DesinfoVacinalModel(
            pretrained_model_name=self.config.pre_trained_model,
            num_labels=self.num_labels,
        ).to(self.device)

    def train(self):
        self.history = []

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.parameters.learning_rate
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader)
            * self.config.parameters.num_epochs,
        )

        for epoch in range(self.config.parameters.num_epochs):
            self.losses = list()
            loss_fn = nn.CrossEntropyLoss()

            self.model.train()
            self.running_loss = 0.0
            
            for batch in self.train_dataloader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                loss = loss_fn(outputs, labels)
                self.losses.append(loss.item())

                loss.backward()

                optimizer.step()
                scheduler.step()
                
                self.running_loss += loss.item()
            
            self.running_loss /= len(self.train_dataset)
            
            self.model.eval()
            self.val_loss = 0.0
            
            with torch.no_grad():
                for batch in self.val_dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                    loss = loss_fn(outputs, labels)
                    self.val_loss += loss.item()
            
            self.val_loss /= len(self.val_dataset)
        
            self.history.append([self.running_loss, self.val_loss])

    def evaluate(self):
        self.model.eval()

        self.predictions = []
        self.probabilities = []
        self.actual_labels = []

        # Enters a context where gradients are not calculated
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Moving the tensors of the batch to the desired device (CPU or GPU)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Calling the forward method of the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Gets the probabilities
                probs = torch.softmax(outputs, dim=1)

                # Gets the predicted class by taking the argmax
                _, preds = torch.max(outputs, dim=1)

                # Move preds and labels to CPU and convert to lists
                self.predictions.extend(preds.cpu().tolist())
                self.probabilities.extend(probs.cpu().tolist())
                self.actual_labels.extend(labels.cpu().tolist())

        fbeta = fbeta_score(
            self.actual_labels,
            self.predictions,
            beta=self.config.parameters.beta,
            average="weighted",
        )

        if fbeta > self.best_fbeta:
            self.best_fbeta = fbeta
            self.save_metrics()

            save_best_model(
                config=self.config,
                model=self.model,
                labels=self.labels,
                curr_seed=self.curr_seed,
                best_fbeta=self.best_fbeta,
                output_path=self.output_path,
            )
            
            save_training_history(
                history=self.history,
                output_path=self.output_path / "metrics",
            )
        

    def save_metrics(self):
        output_metrics_dir = self.output_path / "metrics"
        output_metrics_dir.mkdir(parents=True, exist_ok=True)

        y_true = np.array(self.actual_labels)
        y_pred = np.array(self.predictions)
        y_probs = np.array(self.probabilities)

        is_binary = self.num_labels == 2

        # Saving Confusion Matrix
        save_confusion_matrix(
            output_path=output_metrics_dir,
            y_true=y_true,
            y_pred=y_pred,
            labels=list(range(self.num_labels)),
            display_labels=self.labels,
        )

        # Saving ROC Curve plot
        save_roc_curve(
            output_path=output_metrics_dir,
            y_true=y_true,
            y_probs=y_probs[:, 1] if is_binary else y_probs,
            is_binary=is_binary,
            labels=self.labels,
        )

        # Calculate AUC
        if is_binary:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")

        # Saving Classification Report
        save_metrics_report(
            output_path=output_metrics_dir,
            y_true=y_true,
            y_pred=y_pred,
            beta=self.config.parameters.beta,
            auc=auc,
        )

    def deploy(self):
        """Deploy the best model to MLflow"""

        deploy_run(config=self.config, output_path=self.output_path, device=self.device)
