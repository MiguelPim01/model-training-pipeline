from html import unescape as html_unescape
from pathlib import Path
import unicodedata
import logging
import re

import pandas as pd

from src.domain.pipelines.data_pipeline import IDataPipeline, ProcessedData
from src.infra.schemas.model_config import ModelConfig


class CSVDataPipeline(IDataPipeline):
    """Pipeline for loading and preprocessing CSV text classification data."""

    # Characters to remove (control chars except newline/tab)
    _CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    # Multiple whitespace normalization
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    # URL pattern (http, https, ftp, and www links)
    _URL_PATTERN = re.compile(
        r"(?:https?://|ftp://|www\.)[^\s<>\"\'\)\]]*"
        r"|(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|br|co|uk|de|fr|es|it|pt|ru|cn|jp)[^\s<>\"\'\)\]]*",
        re.IGNORECASE,
    )

    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_dir = Path(config.data.data_dir)

    def load(self) -> ProcessedData:
        """Load and preprocess data from CSV file.

        Returns:
            ProcessedData: Cleaned text samples, encoded labels, and label names.

        Raises:
            RuntimeError: If data file is missing or invalid.
        """
        data_dir = Path(self.config.data.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "train" / "data.csv"

        logging.info(f"Loading data from: {file_path}")

        df = self._load_csv(file_path)
        self._validate_columns(df, file_path)

        initial_count = len(df)
        df = self._handle_missing_values(df)
        df = self._preprocess_text(df)
        df = self._remove_empty_texts(df)
        df = self._remove_duplicates(df)

        self._log_data_statistics(df, initial_count)
        self._check_class_imbalance(df)

        self.X = df["text"].tolist()
        self.labels = sorted(df["label"].unique().tolist())
        label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.y = [label_to_idx[label] for label in df["label"]]

        logging.info(f"Preprocessing completed. Final samples: {len(self.X)}")
        logging.info(f"Labels: {self.labels}")

        return ProcessedData(X=self.X, y=self.y, labels=self.labels)

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            return pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            logging.warning("UTF-8 decode failed, trying latin-1 encoding")
            return pd.read_csv(file_path, encoding="latin-1")
        except FileNotFoundError:
            logging.error(f"Data file not found: {file_path}")
            raise RuntimeError(f"Data file not found: {file_path}")
        except pd.errors.EmptyDataError:
            logging.error(f"Data file is empty: {file_path}")
            raise RuntimeError(f"Data file is empty: {file_path}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise RuntimeError(f"Failed to load data: {e}")

    def _validate_columns(self, df: pd.DataFrame, file_path: Path) -> None:
        """Validate required columns exist."""
        required_cols = {"text", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Missing required columns {missing} in {file_path}. "
                f"Found columns: {list(df.columns)}"
            )

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing text or label values."""
        missing_text = df["text"].isna().sum()
        missing_label = df["label"].isna().sum()

        if missing_text > 0:
            logging.warning(f"Dropping {missing_text} rows with missing text")
        if missing_label > 0:
            logging.warning(f"Dropping {missing_label} rows with missing labels")

        return df.dropna(subset=["text", "label"]).reset_index(drop=True)

    def _preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning transformations."""
        df = df.copy()
        df["text"] = df["text"].astype(str).apply(self._clean_text)
        return df

    def _clean_text(self, text: str) -> str:
        """Clean a single text string.

        Steps:
        1. Unicode normalization (NFC)
        2. HTML entity decoding
        3. URL/link removal
        4. Control character removal
        5. Whitespace normalization
        6. Strip leading/trailing whitespace
        """
        # Unicode normalization (compose characters)
        text = unicodedata.normalize("NFC", text)

        # Decode HTML entities (&amp; -> &, &lt; -> <, etc.)
        text = html_unescape(text)

        # Remove URLs and links
        text = self._URL_PATTERN.sub("", text)

        # Remove control characters (keep newlines and tabs for now)
        text = self._CONTROL_CHAR_PATTERN.sub("", text)

        # Normalize whitespace (multiple spaces/newlines -> single space)
        text = self._WHITESPACE_PATTERN.sub(" ", text)

        # Strip leading/trailing whitespace
        return text.strip()

    def _remove_empty_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where text is empty after cleaning."""
        empty_mask = df["text"].str.strip() == ""
        empty_count = empty_mask.sum()

        if empty_count > 0:
            logging.warning(f"Dropping {empty_count} rows with empty text")
            return df[~empty_mask].reset_index(drop=True)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate text entries (keeping first occurrence)."""
        duplicate_count = df.duplicated(subset=["text"], keep="first").sum()

        if duplicate_count > 0:
            logging.warning(f"Dropping {duplicate_count} duplicate text entries")
            return df.drop_duplicates(subset=["text"], keep="first").reset_index(
                drop=True
            )
        return df

    def _log_data_statistics(self, df: pd.DataFrame, initial_count: int) -> None:
        """Log dataset statistics."""
        final_count = len(df)
        dropped = initial_count - final_count

        logging.info(f"Initial samples: {initial_count}")
        if dropped > 0:
            logging.info(
                f"Dropped samples: {dropped} ({dropped / initial_count * 100:.1f}%)"
            )

        # Text length statistics
        text_lengths = df["text"].str.len()
        logging.info(
            f"Text length stats - "
            f"min: {text_lengths.min()}, "
            f"max: {text_lengths.max()}, "
            f"mean: {text_lengths.mean():.1f}"
        )

    def _check_class_imbalance(self, df: pd.DataFrame) -> None:
        """Warn if significant class imbalance is detected."""
        label_counts = df["label"].value_counts()
        min_count = label_counts.min()
        max_count = label_counts.max()

        # Log class distribution
        logging.info(f"Class distribution:\n{label_counts.to_string()}")

        # Warn if imbalance ratio > 3:1
        if max_count / min_count > 3:
            logging.warning(
                f"Significant class imbalance detected. "
                f"Ratio: {max_count / min_count:.1f}:1. "
                f"Consider using class weights or resampling."
            )
