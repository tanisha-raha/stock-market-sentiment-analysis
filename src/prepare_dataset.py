"""Prepare the Hugging Face Twitter Financial News dataset for this project.

Run from the repository root with ``python -m src.prepare_dataset``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .preprocessing import clean_text

DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
# The repository's dataset card documents these original integer labels:
# LABEL_0 = Bearish, LABEL_1 = Bullish, LABEL_2 = Neutral.
VERIFIED_LABEL_MAPPING: dict[int, str] = {0: "negative", 1: "positive", 2: "neutral"}
REQUIRED_SPLITS = ("train", "validation")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"


def prepare_frame(dataset_splits: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standardise verified source splits to deduplicated ``text``/``sentiment`` rows.

    The source has train and validation but no separate test split. They are combined
    before the existing pipeline makes its own stratified holdout, avoiding duplicate
    text across train/test while keeping the final evaluation unseen by TF-IDF/models.
    """
    missing = [split for split in REQUIRED_SPLITS if split not in dataset_splits]
    if missing:
        raise ValueError(f"Expected dataset split(s) not available: {', '.join(missing)}")
    split_sizes = {split: len(dataset_splits[split]) for split in dataset_splits}
    frames = [dataset_splits[split].to_pandas()[["text", "label"]] for split in REQUIRED_SPLITS]
    frame = pd.concat(frames, ignore_index=True)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must contain text and label features.")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    unknown = sorted(set(labels.dropna().astype(int)) - set(VERIFIED_LABEL_MAPPING))
    if unknown:
        raise ValueError(f"Unexpected source label(s): {unknown}. Re-verify the dataset card mapping.")
    frame = pd.DataFrame({"text": frame["text"].map(clean_text), "sentiment": labels.map(VERIFIED_LABEL_MAPPING)})
    frame = frame.dropna(subset=["sentiment"])
    frame = frame.loc[frame["text"] != ""].drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid rows remain after preparation.")
    return frame, split_sizes


def prepare_dataset(output_path: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Download, inspect, standardise, and write the project training CSV."""
    try:
        from datasets import load_dataset
        source = load_dataset(DATASET_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {DATASET_NAME}. Check internet access and the datasets installation."
        ) from exc
    print(f"Dataset: {DATASET_NAME}")
    print(f"Available original splits: {', '.join(source.keys())}")
    for name, split in source.items():
        print(f"  {name}: {len(split)} rows; features: {split.features}")
    print(f"Verified label mapping: {VERIFIED_LABEL_MAPPING}")
    frame, split_sizes = prepare_frame(source)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Rows after validation and exact-text deduplication: {len(frame)}")
    print("Sentiment class counts:")
    print(frame["sentiment"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0).to_string())
    print(f"Output path: {destination}")
    print("First rows:")
    print(frame.head().to_string(index=False))
    return frame


def main() -> None:
    prepare_dataset()


if __name__ == "__main__":
    main()

