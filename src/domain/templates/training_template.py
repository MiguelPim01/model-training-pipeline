from abc import ABC, abstractmethod

class ITrainingTemplate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass