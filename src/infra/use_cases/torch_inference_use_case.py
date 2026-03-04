from tqdm.auto import tqdm
from pathlib import Path
import logging
import sys

from transformers import BertTokenizer
import pandas as pd
import torch

from src.domain.use_cases.inference_use_case import IInferenceUseCase
from src.infra.schemas.model_config import ModelConfig
from src.utils.mlflow import download_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s]",
    stream=sys.stdout,
)

class TorchInferenceUseCase(IInferenceUseCase):
    """Use case for running inference using a PyTorch model.

    Args:
        IInferenceUseCase (_type_): Interface for inference use cases
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = config.parameters.max_length or 512
        tqdm.pandas()

    def _load_model(self, model_name: str) -> None:
        """Download and load the latest model from MLflow."""
        if self.config.mlflow is None:
            raise ValueError("MLflow configuration is required for inference")

        self.model = download_model(
            tracking_uri=self.config.mlflow.tracking_uri,
            model_name=model_name,
            stage="None",  # Gets the latest version
            device=self.device,
        )
        self.model.eval()

        # Load tokenizer based on the pre-trained model
        self.tokenizer = BertTokenizer.from_pretrained(self.config.pre_trained_model)
        logging.info(f"Tokenizer loaded from {self.config.pre_trained_model}")

    def _predict(self, text: str) -> int:
        """Run inference on a single text.

        Args:
            text: Input text to classify

        Returns:
            Predicted label (0 or 1), or -1 if text is invalid
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before inference")

        # Handle NaN, None, or non-string values
        if not isinstance(text, str) or pd.isna(text):
            logging.warning(f"Invalid text value encountered: {type(text)}. Returning -1.")
            return -1

        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(logits, dim=1).item()

        return prediction

    def _get_csv_files(self, input_path: Path) -> list[Path]:
        """Get all CSV files from the input directory.

        Args:
            input_path: Path to the input directory

        Returns:
            List of CSV file paths
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        csv_files = list(input_path.glob("**/*.csv"))
        if not csv_files:
            logging.warning(f"No CSV files found in {input_path}")

        return csv_files

    def _process_dataset(self, csv_path: Path, output_dir: Path) -> None:
        """Process a single CSV dataset and save labeled results.

        Args:
            csv_path: Path to the input CSV file
            output_dir: Path to the output directory

        Raises:
            ValueError: If the dataset doesn't have a 'text' column
        """
        # Create output filename with _labeled suffix
        output_filename = f"{csv_path.stem}_labeled.csv"
        output_path = output_dir / output_filename

        # Check if labeled dataset already exists
        if output_path.exists():
            logging.info(f"Labeled dataset already exists: {output_path}. Skipping.")
            return

        logging.info(f"Processing dataset: {csv_path}")

        df = pd.read_csv(csv_path)

        if "text" not in df.columns:
            raise ValueError(
                f"Dataset '{csv_path}' does not have a required 'text' column. "
                f"Available columns: {list(df.columns)}"
            )

        # Apply inference to each text
        df["label"] = df["text"].progress_apply(self._predict)

        # Save the labeled dataset
        df.to_csv(output_path, index=False)
        logging.info(f"Labeled dataset saved to: {output_path}")

    def run(self, model_name: str) -> None:
        """Run inference on all CSV files in the configured input directory.

        Processes each CSV file, adds predictions as 'label' column,
        and saves results to the labeled output directory.

        Args:
            model_name: Name of the model to use for inference

        Raises:
            ValueError: If inference configuration is missing
            FileNotFoundError: If input directory doesn't exist
            ValueError: If any dataset is missing the 'text' column
        """
        if self.config.inference is None or self.config.inference.input_data_path is None:
            raise ValueError("Inference input_data_path must be configured")

        input_path = Path(self.config.inference.input_data_path)
        output_dir = input_path.parent.parent / "labeled" / input_path.name

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model from MLflow
        self._load_model(model_name=model_name)

        # Get all CSV files to process
        csv_files = self._get_csv_files(input_path)

        if not csv_files:
            logging.info("No CSV files to process")
            return

        logging.info(f"Found {len(csv_files)} CSV file(s) to process")

        # Process each CSV file
        for csv_path in csv_files:
            self._process_dataset(csv_path, output_dir)

        logging.info(f"Inference completed. Processed {len(csv_files)} file(s)")