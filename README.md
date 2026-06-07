# Model Training Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A modular, extensible pipeline for training classification models with PyTorch and Hugging Face Transformers.

---

## Overview

**Model Training Pipeline** provides a clean, template-based architecture for training and evaluating classification models. It handles the complete ML workflow—from data loading and preprocessing to training, evaluation, and experiment tracking with MLflow.

**Who is this for?**

- ML engineers building classification systems;
- Researchers running reproducible experiments with multiple seeds;
- Teams needing standardized training workflows with experiment tracking.

**Main use-cases:**

- Fine-tuning BERT-based models for text classification;
- Running multiple training runs with different seeds for statistical robustness;
- Tracking experiments and deploying models via MLflow.

---

## Key Features

- **Template-based architecture** — Swap data pipelines, training strategies, and models via clean interfaces
- **Automatic text preprocessing** — Unicode normalization, HTML decoding, URL removal, deduplication
- **MLflow integration** — Log parameters, metrics, artifacts, and deploy models
- **YAML configuration** — Single config file drives the entire training run
- **Metrics & visualization** — ROC curves, confusion matrices, classification reports (JSON)
- **FastAPI training interface** — Front-end requests create PostgreSQL-backed training jobs
- **Docker service images** — Separate training and inference runtime Dockerfiles

**Out of scope:**

- Distributed/multi-GPU training (single GPU only)
- Hyperparameter search (use external tools like Optuna)

---

## Pipeline at a Glance

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. Data    │───▶│  2. Train   │───▶│  3. Eval    │───▶│  4. Deploy  │
│     Load &  │    │     Model   │    │     Metrics │    │     MLflow  │
│  Preprocess │    │             │    │     & Save  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

1. **Data** — Load CSV, clean text (normalize, remove URLs/duplicates), encode labels
2. **Train** — Fine-tune BERT model with AdamW + linear warmup scheduler
3. **Eval** — Compute F-beta, ROC-AUC; save best model based on F-beta score
4. **Deploy** — Log run to MLflow with metrics, artifacts, and model

---

## Project Structure

```
model-training-pipeline/
├── api_training.py             # FastAPI training API backed by PostgreSQL
├── api_inference.py            # FastAPI inference/update API
├── Dockerfile.training         # Training API container image
├── Dockerfile.inference        # Inference API container image
├── docker-compose.yml          # Training and inference service profiles
├── main.py                     # CLI entrypoint
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependencies (reproducibility)
├── config/
│   └── config_example.yaml     # Example configuration
├── data/
│   └── <project>/
│       └── train/
│           └── data.csv        # Training data (text, label columns)
├── models/
│   └── <project>/
│       ├── best_model.pth      # Best model checkpoint
│       └── metrics/
│           ├── classification_report.json
│           ├── confusion_matrix.png
│           └── roc_curve.png
├── downloads/
│   └── training/               # Downloaded datasets for API training jobs
├── src/
│   ├── domain/                 # Abstract interfaces (DDD-style)
│   │   ├── pipelines/          # IDataPipeline, ITrainingPipeline
│   │   ├── strategies/         # ITrainingStrategy
│   │   └── templates/          # ITrainingTemplate
│   ├── infra/                  # Concrete implementations
│   │   ├── classifiers/        # Model architectures (DesinfoVacinalModel)
│   │   ├── datasets/           # PyTorch Dataset classes
│   │   ├── pipelines/          # CSVDataPipeline, NoTestSplitPipeline
│   │   ├── schemas/            # Pydantic config models
│   │   ├── strategies/         # TorchTrainingStrategy
│   │   └── templates/          # DesinfoVacinalTemplate
│   └── utils/                  # Metrics, MLflow helpers, model saving
└── tests/
    └── test_api_training.py    # Training API unit tests
```

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | >= 3.10 |
| OS | Linux, macOS, Windows |
| GPU | Optional (CUDA supported) |

**Core dependencies:**

- `torch >= 2.9`
- `transformers >= 4.57`
- `mlflow >= 3.8`
- `pydantic >= 2.12`
- `pyyaml >= 6.0`
- `fastapi >= 0.128`
- `uvicorn >= 0.40`
- `psycopg[binary] >= 3.3`
- `boto3 >= 1.42`

**Optional:**

- CUDA toolkit for GPU acceleration
- MLflow server for experiment tracking UI

---

## Installation (uv)

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

```bash
# Clone the repository
git clone https://github.com/your-org/model-training-pipeline.git
cd model-training-pipeline

# Create virtual environment and install dependencies from lockfile
uv sync

# Verify installation
uv run python -c "import torch; print(torch.__version__)"
```

**Key uv commands:**

| Command | Description |
|---------|-------------|
| `uv sync` | Install exact versions from `uv.lock` |
| `uv add <pkg>` | Add a dependency and update lockfile |
| `uv run <cmd>` | Run command in the virtual environment |
| `uv lock` | Regenerate lockfile after manual edits |

> The `uv.lock` file ensures every collaborator gets identical dependencies.

---

## Quickstart

Run a complete training pipeline from the CLI:

```bash
# 1. Ensure you have data in the expected location
ls data/desinfo_vacinal/train/data.csv

# 2. Run training with your project config
uv run python main.py --config config/<project>.yaml

# 3. Check outputs
ls models/<project>/
# Expected: best_model.pth, metrics/
```

**Expected output:**

```
[2026-02-18 10:00:00 - INFO] Loaded model configuration: Modelo de classificação... v1.0
[2026-02-18 10:00:01 - INFO] Loading data from: data/<project>/train/data.csv
[2026-02-18 10:00:01 - INFO] Preprocessing completed. Final samples: 1234
Training runs: 100%|████████████████████| 30/30 [15:32<00:00, 31.08s/run]
[2026-02-18 10:15:33 - INFO] Best model saved with F-beta: 0.8721
```

---

## Usage

### CLI Training and Inference

```bash
# Training
uv run python main.py --config <path-to-config.yaml>

# Inference
uv run python main.py --config <path-to-config.yaml> --inference
```

- `--config`: Defines the path to the config file you created.
- `--inference`: Runs the inference path instead of the training path.

### Training API

`api_training.py` is the backend interface intended for front-end training requests. It persists job state in PostgreSQL and returns immediately after accepting a job.

Start it locally:

```bash
DB_URL="postgresql://user:password@localhost:5432/dbname" \
TRAINING_TABLE_NAME="Training" \
uv run python api_training.py
```

Create a new training job:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "by_user": "user@example.com",
    "dataset_url": "s3://bucket/path/to/data.csv",
    "version": "1.0"
  }'
```

Start training from an existing database row:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"id": "existing-training-row-id"}'
```

Check status:

```bash
curl http://localhost:8000/train/status/<training-row-id>
```

The training API accepts one active job globally. It uses PostgreSQL advisory transaction locks and rejects a new request with HTTP 409 when any row is `pending` or `in_progress`. Stale active jobs can be marked failed using `TRAINING_STALE_TIMEOUT_MINUTES`.

### Training API Response Shapes

Accepted jobs return:

```json
{
  "message": "Training process started.",
  "id": "<training row id>",
  "status": "pending"
}
```

Status reads return:

```json
{
  "id": "...",
  "by_user": "...",
  "dataset_url": "...",
  "model_url": "...",
  "status": "...",
  "log": "...",
  "version": "...",
  "createdAt": "...",
  "updatedAt": "..."
}
```

### Inference API

Start the inference/update API locally:

```bash
DB_URL="postgresql://user:password@localhost:5432/dbname" \
uv run python api_inference.py
```

The inference service reads PostgreSQL rows, runs the configured inference workflow, and writes predictions back to the database.

### Environment Variables

| Variable | Service | Required | Default | Description |
|----------|---------|----------|---------|-------------|
| `DB_URL` | Training, inference | Yes | None | PostgreSQL connection URL |
| `TRAINING_TABLE_NAME` | Training | Yes | None | Training table name used by the API |
| `CONFIG_FILE_PATH` | Training, inference | No | `config/desinfo_vacinal.yaml` | Model config path used by API scripts |
| `DOWNLOAD_DIR` | Training | No | `downloads/training` | Directory for resolved API training datasets |
| `ALLOW_LOCAL_DATASET_PATHS` | Training | No | `false` | Allows local dataset paths for development only |
| `TRAINING_STALE_TIMEOUT_MINUTES` | Training | No | `120` | Marks old `pending`/`in_progress` rows failed before accepting a new job |
| `TRAINING_DATA_S3_BUCKET` | Training script | No | None | Existing training dataset bucket used before merging uploaded data |
| `TRAINING_DATA_S3_PREFIX` | Training script | No | Empty | Prefix for existing S3 training CSVs |
| `TRAINING_DATA_S3_REGION` | Training script | No | None | AWS region for S3 dataset access |
| `ALLOW_FIRST_TRAINING_DATASET` | Training script | No | `false` | Allows first training run with only uploaded data when no S3 dataset exists |
| `TIME_RANGE_DAYS` | Inference | No | `365` | Database record time window |
| `UPDATE_INTERVAL_MINUTES` | Inference | No | `120` | Inference update interval |
| `UPDATE_FILE_NAME` | Inference | No | `update_records.json` | Update tracking file name |

### Common Commands

| Task | Command |
|------|---------|
| Train model | `uv run python main.py --config config/desinfo_vacinal.yaml` |
| Start training API | `uv run python api_training.py` |
| Start inference API | `uv run python api_inference.py` |
| Start MLflow Server | `uv run mlflow server --port 5000` |
| View metrics | `cat models/desinfo_vacinal/metrics/classification_report.json` |

---

## Docker

The repository has separate Dockerfiles for the training API and inference API.

### Build Images

```bash
docker build -f Dockerfile.training -t model-training-training .
docker build -f Dockerfile.inference -t model-training-inference .
```

### Run the Training API Container

```bash
docker run --rm \
  --env-file .env \
  -p 8001:8000 \
  -v "$PWD:/app" \
  model-training-training
```

The training API listens on container port `8000`. This example exposes it as `http://localhost:8001` on the host.

Minimum `.env` values for the training API:

```dotenv
DB_URL=postgresql://user:password@host.docker.internal:5432/dbname
TRAINING_TABLE_NAME=Training
CONFIG_FILE_PATH=config/desinfo_vacinal.yaml
```

Use `ALLOW_LOCAL_DATASET_PATHS=true` only for development. Production training requests should use `s3://...` dataset URLs.

### Run the Inference API Container

```bash
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  -v "$PWD:/app" \
  model-training-inference
```

Minimum `.env` values for the inference API:

```dotenv
DB_URL=postgresql://user:password@host.docker.internal:5432/dbname
CONFIG_FILE_PATH=config/desinfo_vacinal.yaml
```

### Docker Compose

Start the training API profile:

```bash
docker compose --profile train up --build desinfo-training-service
```

Start the inference API profile:

```bash
docker compose --profile inf up --build desinfo-inference-service
```

Compose maps:

| Service | Dockerfile | Host URL |
|---------|------------|----------|
| `desinfo-training-service` | `Dockerfile.training` | `http://localhost:8001` |
| `desinfo-inference-service` | `Dockerfile.inference` | `http://localhost:8000` |

The current compose inference service reads `.env`. The current compose training service does not declare `env_file: .env`, so export the required variables before starting it or pass them through your shell environment.

---

## Configuration

Configuration files live in `config/` and use YAML format.

### Example Configuration

```yaml
model_name: Your model name
version: 1.0
description: Your model description
pre_trained_model: neuralmind/bert-base-portuguese-cased

parameters:
  learning_rate: 2e-5      # AdamW learning rate
  batch_size: 16           # Training batch size
  num_epochs: 4            # Epochs per run
  max_length: 512          # Max token sequence length
  beta: 0.5                # F-beta score beta value

data:
  data_dir: data/<project>  # Path to data directory
  test_split: 0.2           # Test split ratio
  val_split: 0.1            # Validation split ratio

mlflow:
  experiment_name: Experiment Name
  tracking_uri: http://localhost:5000
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pre_trained_model` | Hugging Face model ID | Required |
| `parameters.learning_rate` | Optimizer learning rate | `2e-5` |
| `parameters.batch_size` | Samples per batch | `16` |
| `parameters.num_epochs` | Training epochs | `4` |
| `parameters.max_length` | Max sequence length | `512` |
| `data.test_split` | Test set fraction | `0.2` |
| `data.val_split` | Validation set fraction | `0.1` |

---

## Data & Datasets

### Expected Format

Place your data in `data/<project>/train/data.csv` with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Input text to classify |
| `label` | string | Class label |

> If you want to use another dataset configuration you should implement it from the `IDataPipeline` interface.

### Example

```csv
text,label
"Vacinas causam autismo",desinformacao
"Vacinas são seguras e eficazes",informacao
```

### Dataset URLs for API Training

The training API stores `dataset_url` in PostgreSQL and resolves it to a local CSV before calling `training_script.py`.

Supported inputs:

| Input | Support | Notes |
|-------|---------|-------|
| `s3://bucket/key.csv` | Supported | Uses `boto3` to download into `DOWNLOAD_DIR` |
| Local path | Development only | Requires `ALLOW_LOCAL_DATASET_PATHS=true` |
| `http://` or `https://` | Not currently supported | Add explicit project support before using public URLs |

Resolved files must exist, be non-empty, have a `.csv` extension, and include `text` and `label` columns.

### Preprocessing Pipeline

The `CSVDataPipeline` automatically:

1. **Handles encoding** — Falls back to latin-1 if UTF-8 fails
2. **Normalizes Unicode** — NFC normalization for consistent characters
3. **Decodes HTML entities** — `&amp;` → `&`, `&lt;` → `<`
4. **Removes URLs** — Strips http/https/ftp links and domain patterns
5. **Removes duplicates** — Keeps first occurrence only
6. **Validates data** — Logs warnings for missing values, class imbalance

---

## Training

### Starting a Run

```bash
uv run python main.py --config config/<project>.yaml
```

The `NoTestSplitPipeline` pipeline runs 30 training iterations with different random seeds to ensure statistical robustness. The best model (highest F-beta score) is saved automatically.

### Training Details

- **Optimizer:** AdamW with linear warmup scheduler
- **Loss:** CrossEntropyLoss
- **Runs:** 30 seeds (reproducible, hardcoded)
- **Checkpointing:** Best model saved to `models/<project>/best_model.pth`

### Hardware Notes

- **CPU:** Works but slow (~30 min per run)
- **GPU:** Recommended, auto-detected via `torch.cuda.is_available()`

---

## Evaluation

### Metrics Reported

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision | Per-class and macro precision |
| Recall | Per-class and macro recall |
| F1-score | Per-class and macro F1 |
| F-beta | Weighted harmonic mean (configurable beta) |
| ROC-AUC | Area under ROC curve |

### Output Artifacts

```
models/<project>/metrics/
├── classification_report.json   # Full metrics report
├── confusion_matrix.png         # Confusion matrix visualization
└── roc_curve.png                # ROC curve (per class for multiclass)
```

### Reproducing Evaluation

To reproduce metrics for a saved checkpoint, load the model and run validation:

```python
from src.infra.classifiers.desinfo_vacinal_model import DesinfoVacinalModel
import torch

model = DesinfoVacinalModel(pretrained_model_name="...", num_labels=2)
model.load_state_dict(torch.load("models/desinfo_vacinal/best_model.pth"))
model.eval()
# Run inference on validation set
```

---

## Experiment Tracking & Logs

### MLflow Integration

When `mlflow` is configured, the pipeline logs:

| What | Where |
|------|-------|
| Parameters | Learning rate, batch size, epochs, etc. |
| Metrics | F-beta, accuracy, precision, recall, ROC-AUC |
| Artifacts | Model checkpoint, ROC curve, confusion matrix, config |

### Viewing Runs

```bash
# Start MLflow UI
uv run mlflow ui --port 5000

# Open in browser
open http://localhost:5000
```

### Run Naming

Runs are named `model_v<version>` (e.g., `model_v1.0`). Version auto-increments.

---

## Reproducibility

### Ensuring Reproducible Results

1. **Fixed seeds** — 30 hardcoded seeds in `NoTestSplitPipeline._get_seeds()`
2. **Locked dependencies** — `uv.lock` pins exact package versions
3. **Config versioning** — Store config YAML alongside results
4. **PyTorch seed** — `torch.manual_seed(1)` set in strategy

### Sharing Reproducible Runs

To share results for exact reproduction:

```
1. uv.lock file (exact dependencies)
2. config/*.yaml used for training
3. data/<project>/train/data.csv (or data hash)
4. MLflow run ID or experiment name
```

---

## Extending the Pipeline

### Adding a New Model

1. Create classifier in `src/infra/classifiers/`:

```python
# src/infra/classifiers/my_model.py
from torch import nn

class MyModel(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int):
        super().__init__()
        # Define architecture
    
    def forward(self, input_ids, attention_mask):
        # Forward pass
        return logits
```

2. Update strategy to use your model in `build_model()`

### Adding New Data

1. Implement `IDataPipeline` interface:

```python
# src/infra/pipelines/my_pipeline.py
from src.domain.pipelines.data_pipeline import IDataPipeline, ProcessedData

class MyDataPipeline(IDataPipeline):
    def load(self) -> ProcessedData:
        # Load and preprocess data
        return ProcessedData(X=texts, y=labels, labels=label_names)
```

2. Use in your template

> You can also implement your own training strategies, templates and pipelines extending the interfaces needed.

### Required Interfaces

| Interface | Methods |
|-----------|---------|
| `IDataPipeline` | `load() -> ProcessedData` |
| `ITrainingStrategy` | `load_dataset()`, `load_datamodel()`, `build_model()`, `train()`, `evaluate()`, `save_metrics()`, `deploy()` |
| `ITrainingPipeline` | `run()` |
| `ITrainingTemplate` | `run()` |

---

## Testing & Quality

### Running Tests

```bash
# Run the current unit tests
.venv/bin/python -m unittest tests.test_api_training

# Or through uv when the environment is synced
uv run python -m unittest tests.test_api_training
```

The project currently uses standard-library `unittest` for the training API tests. `pytest` is not declared as a project dependency.

### Linting & Formatting

```bash
# Check code style
uvx ruff check src

# Auto-fix issues
uvx ruff check src --fix

# Format code
uvx ruff format src
```

### Pre-commit Hooks (recommended)

```bash
# Install pre-commit
uv add --dev pre-commit

# Setup hooks
uv run pre-commit install
```

---

## Contributing

### How to Contribute

1. **Open an issue** — Describe the bug or feature request
2. **Fork & branch** — Create a feature branch from `main`
3. **Make changes** — Follow the coding style (ruff formatted)
4. **Test** — Ensure tests pass locally
5. **Submit PR** — Reference the issue in your PR description

### Coding Style

- Use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Follow existing patterns in `src/domain/` for interfaces
- Add docstrings to public methods
- Type hints are encouraged

### Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Third-party Licenses

- [PyTorch](https://github.com/pytorch/pytorch) — BSD-style
- [Transformers](https://github.com/huggingface/transformers) — Apache 2.0
- [MLflow](https://github.com/mlflow/mlflow) — Apache 2.0

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{model_training_pipeline,
  title = {Model Training Pipeline},
  author = {Miguel Vieira Machado Pim},
  year = {2026},
  url = {https://github.com/MiguelPim01/model-training-pipeline}
}
```

For the pre-trained Portuguese BERT model:

```bibtex
@inproceedings{souza2020bertimbau,
  title={BERTimbau: Pretrained BERT Models for Brazilian Portuguese},
  author={Souza, F{\'a}bio and Nogueira, Rodrigo and Lotufo, Roberto},
  booktitle={Brazilian Conference on Intelligent Systems},
  year={2020}
}
```

---

## Support / Contact

### Response Expectations

- Issues are triaged within 1 week
- PRs reviewed within 2 weeks

### Maintainers

- **Miguel Vieira Machado Pim** — [@MiguelPim01](https://github.com/MiguelPim01)
