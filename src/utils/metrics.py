from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score
)

from sklearn.preprocessing import label_binarize

from typing import List

def save_roc_curve(
    output_path: str, 
    y_true, y_probs,
    is_binary: bool,
    labels: List[str] = None,
    filename: str = "roc_curve.png"
):
    out = Path(output_path)
    if out.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        out_file = out
    else:
        out_file = out / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    if is_binary:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Multi-class: One-vs-Rest ROC curves
        n_classes = len(labels)
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(out_file, dpi=300)
    plt.close()

def save_confusion_matrix(
    output_path: str, 
    y_true, y_pred, 
    labels: List[int] = None,
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