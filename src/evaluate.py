"""Model metrics and visualisation helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, precision_recall_fscore_support

LABELS = ["negative", "neutral", "positive"]


def evaluate_predictions(y_true, y_pred, model_name: str, figures_dir: str | Path | None = None) -> dict:
    """Calculate consistent classification metrics and optionally save a matrix."""
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {"Model": model_name, "Accuracy": accuracy_score(y_true, y_pred), "Macro F1": macro_f1,
               "Weighted F1": weighted_f1, "Precision": precision, "Recall": recall,
               "classification_report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)}
    if figures_dir is not None:
        path = Path(figures_dir); path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{model_name} confusion matrix")
        fig.tight_layout(); fig.savefig(path / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150); plt.close(fig)
    return metrics


def save_comparison(metrics: list[dict], output_path: str | Path) -> pd.DataFrame:
    comparison = pd.DataFrame(metrics).drop(columns=["classification_report"], errors="ignore")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def add_to_comparison(metrics: dict, output_path: str | Path) -> pd.DataFrame:
    """Add or replace one model row without altering existing model results."""
    path = Path(output_path)
    new_row = pd.DataFrame([metrics]).drop(columns=["classification_report"], errors="ignore")
    if path.exists():
        comparison = pd.read_csv(path)
        comparison = comparison.loc[comparison["Model"] != metrics["Model"]]
        comparison = pd.concat([comparison, new_row], ignore_index=True)
    else:
        comparison = new_row
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)
    return comparison


def plot_model_comparison(comparison: pd.DataFrame, figures_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]].plot.bar(ax=ax)
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Model performance comparison")
    fig.tight_layout(); Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / "model_comparison.png", dpi=150); plt.close(fig)


def plot_sentiment_distribution(frame: pd.DataFrame, figures_dir: str | Path) -> None:
    """Save the labelled class distribution used for a training experiment."""
    fig, ax = plt.subplots(figsize=(6, 4))
    frame["sentiment"].value_counts().reindex(LABELS, fill_value=0).plot.bar(ax=ax, color=["#d95f02", "#7570b3", "#1b9e77"])
    ax.set(title="Sentiment class distribution", xlabel="Sentiment", ylabel="Posts")
    fig.tight_layout(); Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / "sentiment_class_distribution.png", dpi=150); plt.close(fig)
