from abc import ABC, abstractmethod

class TrainingPipelineTemplate(ABC):
    def __init__(self):
        pass

    def run(self):
        self.preprocess_data()
        self.load_dataset()
        self.load_datamodel()
        self.train_model()
        self.evaluate_model()
        self.deploy_model()
    
    @abstractmethod
    def preprocess_data(self):
        pass
    
    
    def load_dataset(self):
        pass
    
    
    def load_datamodel(self):
        pass
    
    @abstractmethod
    def train_model(self):
        pass
    
    def evaluate_model(self):
        pass
    
    def deploy_model(self):
        pass