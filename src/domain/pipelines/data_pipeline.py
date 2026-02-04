from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class ProcessedData:
    X: List[str]
    y: List[int]
    labels: List[str]

class IDataPipeline(ABC):
    
    @abstractmethod
    def load(self) -> ProcessedData:
        """Load and preprocess data, returning processed data."""
        pass