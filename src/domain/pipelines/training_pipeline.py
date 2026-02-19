from abc import ABC, abstractmethod

from src.domain.strategies.training_strategy import ITrainingStrategy
from src.domain.pipelines.data_pipeline import IDataPipeline


class ITrainingPipeline(ABC):
    """Generic Pipeline for training models"""

    def __init__(self, strategy: ITrainingStrategy, data_pipeline: IDataPipeline):
        self.strategy = strategy
        self.data_pipeline = data_pipeline

    @abstractmethod
    def run(self):
        pass
