from abc import ABC, abstractmethod

class IInferenceUseCase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, data):
        pass