"""CSV loading, column standardisation, and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .preprocessing import clean_text, normalize_label

STANDARD_COLUMNS = ("text", "sentiment", "date", "ticker")


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot support the requested analysis."""


def load_dataset(path: str | Path, column_map: Mapping[str, str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load a local CSV and return cleaned standard columns plus warnings.

    ``column_map`` maps standard names (for example ``text``) to CSV names.
    Text and sentiment are required; date and ticker are optional.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetValidationError(f"CSV not found: {file_path}")
    try:
        frame = pd.read_csv(file_path)
    except Exception as exc:
        raise DatasetValidationError(f"Could not read CSV: {exc}") from exc
    if frame.empty:
        raise DatasetValidationError("The CSV contains no rows.")
    mapping = dict(column_map or {})
    rename = {mapping.get(name, name): name for name in STANDARD_COLUMNS if mapping.get(name, name) in frame.columns}
    frame = frame.rename(columns=rename)
    missing = [name for name in ("text", "sentiment") if name not in frame.columns]
    if missing:
        raise DatasetValidationError("Missing required column(s): " + ", ".join(missing))
    warnings: list[str] = []
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean_text)
    empty_text = int((frame["text"] == "").sum())
    if empty_text:
        warnings.append(f"Dropped {empty_text} rows with empty text.")
        frame = frame.loc[frame["text"] != ""].copy()
    labels = frame["sentiment"].map(normalize_label)
    unsupported = int(labels.isna().sum())
    if unsupported:
        warnings.append(f"Dropped {unsupported} rows with missing or unsupported sentiment labels.")
        frame = frame.loc[labels.notna()].copy()
        labels = labels.loc[labels.notna()]
    frame["sentiment"] = labels
    duplicate_count = int(frame.duplicated(subset=["text", "sentiment"]).sum())
    if duplicate_count:
        warnings.append(f"Dropped {duplicate_count} duplicate text/label rows.")
        frame = frame.drop_duplicates(subset=["text", "sentiment"]).copy()
    if frame.empty:
        raise DatasetValidationError("No valid rows remain after validation.")
    if "date" in frame.columns:
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        bad_dates = int(parsed.isna().sum())
        if bad_dates:
            warnings.append(f"{bad_dates} malformed/missing dates retained as missing; market analysis excludes them.")
        frame["date"] = parsed
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
        frame.loc[frame["ticker"] == "", "ticker"] = pd.NA
    return frame, warnings


def require_market_columns(frame: pd.DataFrame) -> None:
    """Confirm usable date and ticker columns before market analysis."""
    missing = [col for col in ("date", "ticker") if col not in frame.columns]
    if missing:
        raise DatasetValidationError(
            "Market analysis cannot run without date and ticker columns. Missing: " + ", ".join(missing)
        )
    if frame[["date", "ticker"]].dropna().empty:
        raise DatasetValidationError("Market analysis cannot run: date/ticker values are missing or malformed.")

