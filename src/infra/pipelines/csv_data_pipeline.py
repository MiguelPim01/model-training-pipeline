from pathlib import Path
import pandas as pd
import logging

from src.domain.pipelines.data_pipeline import IDataPipeline, ProcessedData
from src.infra.schemas.model_config import ModelConfig


class CSVDataPipeline(IDataPipeline):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_dir = Path(config.data.data_dir)

    def load(self) -> ProcessedData:
        data_dir = Path(self.config.data.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

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

        self.labels = list(df["label"].value_counts().index)

        self.y = [self.labels.index(label) for label in df["label"]]

        logging.info(f"Data preprocessing completed. Number of samples: {len(self.X)}")
        logging.info(f"Labels found: {self.labels}")

        return ProcessedData(X=self.X, y=self.y, labels=self.labels)
