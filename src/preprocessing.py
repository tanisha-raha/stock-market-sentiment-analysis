"""Text and label preprocessing utilities."""
from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

LABEL_MAP = {
    "positive": "positive", "pos": "positive", "bullish": "positive", "1": "positive", 1: "positive", 1.0: "positive",
    "neutral": "neutral", "neu": "neutral", "0": "neutral", 0: "neutral", 0.0: "neutral",
    "negative": "negative", "neg": "negative", "bearish": "negative", "-1": "negative", -1: "negative", -1.0: "negative",
}


def clean_text(value: Any) -> str:
    """Clean social-media text while retaining finance-relevant words."""
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\brt\s+", " ", text)
    # A common RT pattern leaves a leading colon after its author mention.
    text = re.sub(r"^\s*:\s*", "", text)
    text = text.replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value: Any) -> str | None:
    """Map common numeric/text labels to positive, neutral, or negative."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().lower()
    return LABEL_MAP.get(value)
