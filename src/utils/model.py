from pathlib import Path
import tempfile
import shutil
import torch
import os

from typing import List

from src.infra.schemas.model_config import ModelConfig


def save_best_model(
    config: ModelConfig,
    model: torch.nn.Module,
    labels: List[str],
    curr_seed: int,
    best_fbeta: float,
    output_path: Path,
):
    """Save the best model to disk atomically"""

    def atomic_save(checkpoint, path):
        """Avoids corrupted file in case of failure: writes tmp and does atomic rename"""
        d = os.path.dirname(path)

        with tempfile.NamedTemporaryFile(delete=False, dir=d) as f:
            tmp = f.name

        torch.save(checkpoint, tmp)
        shutil.move(tmp, path)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "bert_model_name": config.pre_trained_model,
        "num_classes": len(labels),
        "tokenizer_name": config.pre_trained_model,
        "max_length": config.parameters.max_length,
        "seed": int(curr_seed),
        "fbeta": float(best_fbeta),
    }

    atomic_save(checkpoint, output_path / "best_model.pth")
