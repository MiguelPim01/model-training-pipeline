from pydantic import BaseModel
from typing import Optional
import yaml
from pathlib import Path

from yaml import YAMLError

class ModelParameters(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int
    max_length: Optional[int] = None
    beta: Optional[float] = None
    
class ModelData(BaseModel):
    data_dir: str
    val_split: float

class ModelMLFlow(BaseModel):
    experiment_name: Optional[str] = None
    tracking_uri: Optional[str] = None

class ModelConfig(BaseModel):
    model_name: str
    version: Optional[float] = None
    description: Optional[str] = None
    pre_trained_model: Optional[str] = None
    parameters: ModelParameters
    data: ModelData
    mlflow: Optional[ModelMLFlow] = None


def parse_file(config_file_path: str | Path) -> ModelConfig:
    """Transforms the configuration file into the class ModelConfig

    Args:
        config_file_path (str | Path): Path to the configuration file.

    Raises:
        RuntimeError: File not found
        RuntimeError: Permission denied when accessing the file
        RuntimeError: OS error occurred when accessing the file
        RuntimeError: Error parsing YAML file
        RuntimeError: Error reading configuration file
        RuntimeError: Configuration file is empty
        RuntimeError: Config must be a mapping/dict

    Returns:
        ModelConfig: Parsed model configuration object.
    """
    try:
        with open(config_file_path, 'r') as f:
            config_file = yaml.safe_load(f)
            
    # Checking for file-related errors
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found: {config_file_path}")
    except PermissionError:
        raise RuntimeError(f"Permission denied when accessing the file: {config_file_path}")
    except OSError as e:
        raise RuntimeError(f"OS error occurred when accessing the file: {e}")
    except YAMLError as e:
        raise RuntimeError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading configuration file: {e}")
    
    # Checking for content-related errors
    if config_file is None:
        raise RuntimeError(f"Configuration file is empty: {config_file_path}")
    
    if not isinstance(config_file, dict):
        raise RuntimeError(f"Config must be a mapping/dict, got {type(config_file).__name__}")
    
    return ModelConfig(**config_file)