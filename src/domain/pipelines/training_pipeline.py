from abc import ABC, abstractmethod

from domain.strategies.training_strategy import TrainingStrategy

class ITrainingPipeline(ABC):
    """Generic Pipeline for training models
    """
    
    def __init__(self, strategy: TrainingStrategy):
        self.strategy = strategy
    
    @abstractmethod
    def run(self):
        pass