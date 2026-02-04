from abc import ABC, abstractmethod

from src.infra.schemas.model_config import ModelConfig

class ITrainingStrategy(ABC):
    """Training interface

    Args:
        ABC (_type_): Abstract Base Class
    """
    def __init__(self, config: ModelConfig):
        self.config = config
    
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
    def train(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def save_metrics(self):
        pass
    
    @abstractmethod
    def deploy(self):
        pass