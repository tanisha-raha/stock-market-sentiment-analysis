"""Non-UI helpers for the live Streamlit dashboard."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterable

import joblib
import pandas as pd

from .preprocessing import clean_text

VALID_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
SENTIMENTS = ("positive", "neutral", "negative")


def normalize_ticker(value: str) -> str | None:
    """Return a Yahoo-style ticker or ``None`` for invalid user input."""
    ticker = str(value or "").strip().upper()
    return ticker if VALID_TICKER.fullmatch(ticker) else None


def extract_news_fields(item: dict[str, Any]) -> dict[str, str | None]:
    """Extract common fields from yfinance's legacy and newer news shapes."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = content.get("title") or item.get("title")
    provider = content.get("provider")
    publisher = (provider.get("displayName") if isinstance(provider, dict) else None) or content.get("publisher") or item.get("publisher")
    raw_time = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
    date = None
    if isinstance(raw_time, (int, float)):
        date = datetime.fromtimestamp(raw_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elif raw_time:
        date = str(raw_time)
    link = content.get("canonicalUrl") or content.get("clickThroughUrl") or content.get("url") or item.get("link")
    if isinstance(link, dict):
        link = link.get("url")
    return {"headline": str(title).strip() if title else None, "publisher": str(publisher).strip() if publisher else None, "date": date, "link": str(link) if link else None}


def load_logistic_artifacts(models_dir: str | Path = "models"):
    """Load saved Logistic Regression and vectorizer once per Streamlit process."""
    base = Path(models_dir)
    model_path = base / "logistic_regression.joblib"
    vectorizer_path = base / "tfidf_vectorizer.joblib"
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Saved Logistic Regression model/vectorizer not found. Run baseline training first.")
    return joblib.load(model_path), joblib.load(vectorizer_path)


def classify_headlines(headlines: Iterable[str], model, vectorizer) -> list[dict[str, Any]]:
    """Classify non-empty headlines in one vectorizer/model batch."""
    cleaned = [clean_text(text) for text in headlines]
    cleaned = [text for text in cleaned if text]
    if not cleaned:
        return []
    vectors = vectorizer.transform(cleaned)
    labels = model.predict(vectors)
    probabilities = model.predict_proba(vectors) if hasattr(model, "predict_proba") else None
    results = []
    for index, (text, label) in enumerate(zip(cleaned, labels)):
        row: dict[str, Any] = {"cleaned_text": text, "sentiment": str(label)}
        if probabilities is not None:
            row["probabilities"] = {str(name): float(value) for name, value in zip(model.classes_, probabilities[index])}
        results.append(row)
    return results


def aggregate_sentiment(labels: Iterable[str]) -> dict[str, float | int | str]:
    """Summarise headline labels into proportions, index, and cautious wording."""
    values = [str(label).lower() for label in labels if str(label).lower() in SENTIMENTS]
    counts = {label: values.count(label) for label in SENTIMENTS}
    total = len(values)
    score = (counts["positive"] - counts["negative"]) / total if total else 0.0
    if score >= 0.5: overall = "Strongly Positive"
    elif score >= 0.15: overall = "Positive"
    elif score <= -0.5: overall = "Strongly Negative"
    elif score <= -0.15: overall = "Negative"
    else: overall = "Neutral"
    return {**counts, "total": total, "positive_percentage": 100 * counts["positive"] / total if total else 0.0,
            "neutral_percentage": 100 * counts["neutral"] / total if total else 0.0,
            "negative_percentage": 100 * counts["negative"] / total if total else 0.0,
            "sentiment_index": score, "overall_sentiment": overall}


def behavioral_signal(sentiment_index: float, buy_threshold: float = 0.25, sell_threshold: float = -0.25) -> str:
    """Return a non-predictive experimental behavioural signal from present sentiment."""
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must exceed sell_threshold.")
    if sentiment_index >= buy_threshold: return "BUY"
    if sentiment_index <= sell_threshold: return "SELL"
    return "HOLD"

