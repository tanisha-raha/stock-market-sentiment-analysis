"""Pretrained, batched ProsusAI/finbert inference and held-out evaluation."""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import pandas as pd

from .evaluate import evaluate_predictions

MODEL_NAME = "ProsusAI/finbert"
EXPECTED_LABELS = {"negative", "neutral", "positive"}


def resolve_finbert_labels(id2label: Mapping[int | str, str]) -> dict[int, str]:
    """Validate and normalise the three semantic labels exposed by FinBERT's config."""
    labels = {int(index): str(label).strip().lower() for index, label in id2label.items()}
    if set(labels.values()) != EXPECTED_LABELS or len(labels) != 3:
        raise RuntimeError(
            f"Unexpected {MODEL_NAME} label configuration: {labels}. "
            "Expected negative, neutral, and positive."
        )
    return labels


def _preferred_device(torch) -> str:
    """Prefer CUDA; use MPS only when PyTorch reports it available, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def _batches(values: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    iterator = iter(values)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def predict_finbert(texts: Iterable[str], batch_size: int = 16) -> list[str]:
    """Return FinBERT predictions using inference only, with safe MPS fallback."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
        label_map = resolve_finbert_labels(model.config.id2label)
    except Exception as exc:
        raise RuntimeError("FinBERT could not be loaded. Check internet access and transformers/PyTorch installation.") from exc
    device = _preferred_device(torch)
    try:
        model = model.to(device)
    except RuntimeError:
        if device != "mps": raise
        device = "cpu"; model = model.to(device)
    predictions: list[str] = []
    for batch in _batches(texts, batch_size):
        try:
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.inference_mode(): indices = model(**encoded).logits.argmax(dim=1).cpu().tolist()
        except RuntimeError:
            if device != "mps": raise
            # A failed MPS operation is retried on CPU for this and all later batches.
            device = "cpu"; model = model.to(device)
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.inference_mode(): indices = model(**encoded).logits.argmax(dim=1).cpu().tolist()
        predictions.extend(label_map[int(index)] for index in indices)
    return predictions


def evaluate_finbert(x_test: pd.Series, y_test: pd.Series, figures_dir: str | Path = "outputs/figures", metrics_dir: str | Path = "outputs/metrics", batch_size: int = 16) -> dict:
    """Evaluate pretrained FinBERT on supplied held-out texts and save actual metrics."""
    metrics = evaluate_predictions(y_test, predict_finbert(x_test.tolist(), batch_size=batch_size), "FinBERT", figures_dir)
    path = Path(metrics_dir); path.mkdir(parents=True, exist_ok=True)
    with (path / "finbert_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics
