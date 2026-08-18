"""Daily sentiment aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

SCORES = {"positive": 1, "neutral": 0, "negative": -1}


def daily_sentiment_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post labels by date and ticker, including optional engagement weight."""
    required = {"date", "ticker", "sentiment"}
    if not required.issubset(frame.columns): raise ValueError("Daily sentiment index requires date, ticker, and sentiment columns.")
    data = frame.dropna(subset=["date", "ticker"]).copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize(); data["score"] = data["sentiment"].map(SCORES)
    if data["score"].isna().any(): raise ValueError("Unsupported sentiment label found.")
    for column in ("likes", "favorites", "retweets", "reposts"):
        if column in data: data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).clip(lower=0)
    likes = data.get("likes", data.get("favorites", pd.Series(0, index=data.index)))
    reposts = data.get("retweets", data.get("reposts", pd.Series(0, index=data.index)))
    data["engagement_weight"] = 1 + np.log1p(likes + 2 * reposts)
    rows = []
    for (date, ticker), group in data.groupby(["date", "ticker"], sort=True):
        counts = group["sentiment"].value_counts()
        pos, neu, neg = (int(counts.get(label, 0)) for label in ("positive", "neutral", "negative"))
        total = len(group)
        rows.append({"date": date, "ticker": ticker, "mean_sentiment": group["score"].mean(), "positive_count": pos,
                     "neutral_count": neu, "negative_count": neg, "total_posts": total, "positive_proportion": pos / total,
                     "neutral_proportion": neu / total, "negative_proportion": neg / total,
                     "sentiment_index": (pos - neg) / total, "engagement_weighted_sentiment": np.average(group["score"], weights=group["engagement_weight"])})
    return pd.DataFrame(rows)

