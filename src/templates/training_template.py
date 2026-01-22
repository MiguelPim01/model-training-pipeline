from abc import ABC, abstractmethod

class TrainingTemplate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass