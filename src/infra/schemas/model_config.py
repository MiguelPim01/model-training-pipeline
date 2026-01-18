from pydantic import BaseModel
from typing import Optional

class ModelParameters(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int
    max_length: Optional[int] = None

class ModelConfig(BaseModel):
    model_name: str
    version: str
    parameters: ModelParameters
    pre_trained_model: Optional[str] = None
    description: Optional[str] = None