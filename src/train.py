"""Leakage-safe TF-IDF baseline training and saved-model inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .evaluate import evaluate_predictions, plot_model_comparison, save_comparison
from .preprocessing import clean_text


def split_data(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    y = frame["sentiment"]
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    try:
        return train_test_split(frame["text"], y, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError as exc:
        raise ValueError("Dataset is too small for an 80/20 split; add more labelled rows.") from exc


def train_baselines(frame: pd.DataFrame, models_dir: str | Path = "models", metrics_dir: str | Path = "outputs/metrics", figures_dir: str | Path = "outputs/figures", vectorizer_params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, tuple[pd.Series, pd.Series]]:
    """Split first, fit TF-IDF only on training text, then train two models."""
    x_train, x_test, y_train, y_test = split_data(frame)
    params = {"max_features": 10000, "ngram_range": (1, 2), "min_df": 2, "stop_words": "english"}
    params.update(vectorizer_params or {})
    if len(x_train) < 2: raise ValueError("Need at least two training rows.")
    vectorizer = TfidfVectorizer(**params)
    try: x_train_vec = vectorizer.fit_transform(x_train)
    except ValueError:
        params["min_df"] = 1; vectorizer = TfidfVectorizer(**params); x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    models = {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)}
    Path(models_dir).mkdir(parents=True, exist_ok=True); joblib.dump(vectorizer, Path(models_dir) / "tfidf_vectorizer.joblib")
    results = []
    for name, model in models.items():
        model.fit(x_train_vec, y_train); prediction = model.predict(x_test_vec)
        results.append(evaluate_predictions(y_test, prediction, name, figures_dir))
        joblib.dump(model, Path(models_dir) / f"{name.lower().replace(' ', '_')}.joblib")
    comparison = save_comparison(results, Path(metrics_dir) / "model_comparison.csv")
    plot_model_comparison(comparison, figures_dir)
    return comparison, (x_test, y_test)


def predict_sentiment(text: str, model_name: str = "logistic_regression", models_dir: str | Path = "models") -> dict:
    if not clean_text(text): raise ValueError("Prediction text is empty.")
    base = Path(models_dir); vectorizer = joblib.load(base / "tfidf_vectorizer.joblib")
    model = joblib.load(base / f"{model_name}.joblib")
    vector = vectorizer.transform([clean_text(text)]); result = {"sentiment": str(model.predict(vector)[0])}
    if hasattr(model, "predict_proba"):
        result["probabilities"] = {str(k): float(v) for k, v in zip(model.classes_, model.predict_proba(vector)[0])}
    return result

