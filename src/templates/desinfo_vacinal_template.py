import logging
import sys
import pandas as pd
from pathlib import Path

from src.templates.training_pipeline_template import TrainingPipelineTemplate
from src.infra.schemas.model_config import ModelConfig

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class DesinfoVacinalTemplate(TrainingPipelineTemplate):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def preprocess_data(self):
        data_dir = Path(self.config.data.data_dir)
        
        logging.info(f"Preprocessing data from directory: {data_dir}")
        
        try:
            df = pd.read_csv(data_dir / "train" / "data.csv")
        except FileNotFoundError:
            logging.error(f"Data file not found in {data_dir / 'train' / 'data.csv'}")
            raise RuntimeError("Data preprocessing failed due to missing file.")
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise RuntimeError("Data preprocessing failed due to an unexpected error.")
        
        self.X = [str(x) for x in df["text"]]
        
        labels = list(df["label"].value_counts().index)
        self.y = [i for i, _ in enumerate(labels)]
        
        logging.info(f"Data preprocessing completed. Number of samples: {len(self.X)}")
        logging.info(f"Labels found: {labels} - {self.y}")

    def load_dataset(self):
        pass

    def load_datamodel(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass
    
    def deploy_model(self):
        pass