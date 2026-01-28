from abc import ABC, abstractmethod

from src.domain.strategies.training_strategy import ITrainingStrategy

class ITrainingPipeline(ABC):
    """Generic Pipeline for training models
    """
    
    def __init__(self, strategy: ITrainingStrategy):
        self.strategy = strategy
    
    @abstractmethod
    def run(self):
        pass