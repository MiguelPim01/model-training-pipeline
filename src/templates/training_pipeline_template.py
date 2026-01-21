from abc import ABC, abstractmethod

class TrainingPipelineTemplate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass