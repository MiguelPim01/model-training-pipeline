from abc import ABC, abstractmethod
import os
import shutil
import tempfile
import torch

from src.infra.schemas.model_config import ModelConfig

class TrainingStrategy(ABC):
    """Training interface

    Args:
        ABC (_type_): Abstract Base Class
    """
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def preprocess_data(self):
        pass
    
    @abstractmethod
    def load_dataset(self):
        pass
    
    @abstractmethod
    def load_datamodel(self):
        pass
    
    @abstractmethod
    def build_model(self):
        pass
    
    @abstractmethod
    def train(self, epoch: int):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def save_metrics(self):
        pass