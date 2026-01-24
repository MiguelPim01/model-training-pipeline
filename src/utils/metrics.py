from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.metrics import (
    RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score
)

from typing import List

def save_roc_curve(
    output_path: str, 
    y_true, y_pred,
    filename: str = "roc_curve.png"
):
    out = Path(output_path)
    if out.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        out_file = out
    else:
        out_file = out / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(
        y_true,
        y_pred,
        name="ROC curve",
        ax=ax,
        plot_chance_level=True,
        despine=True,
    )
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Receiver Operating Characteristic (ROC) Curve")
    
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    
    plt.close(fig)

def save_confusion_matrix(
    output_path: str, 
    y_true, y_pred, 
    labels: List[str] = None,
    display_labels: List[str] = None,
    filename: str = "confusion_matrix.png"
):
    out = Path(output_path)
    if out.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        out_file = out
    else:
        out_file = out / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix and metrics saving
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=display_labels if display_labels is not None else labels
    )
    
    # Saving Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    display.plot(ax=ax, values_format='d')
    
    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    
    plt.close(fig)

def save_metrics_report(
    output_path: str,
    y_true, y_pred,
    beta: float = None,
    auc: float = None,
    filename: str = "classification_report.json"
):
    out = Path(output_path)
    if out.suffix.lower() in {".txt", ".md", ".json", ".yaml", ".yml"}:
        out_file = out
    else:
        out_file = out / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted')),
        "recall": float(recall_score(y_true, y_pred, average='weighted')),
        "f1_score": float(f1_score(y_true, y_pred, average='weighted')),
        "fbeta_score": float(fbeta_score(y_true, y_pred, beta=beta, average='weighted')) if beta is not None else None,
        "auc": float(auc) if auc is not None else None
    }
    
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)