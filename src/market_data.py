"""Market download, return construction, and date alignment."""
from __future__ import annotations

import pandas as pd


def fetch_market_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Download OHLCV data with yfinance and derive forward returns."""
    try:
        import yfinance as yf
        frame = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    except Exception as exc: raise RuntimeError(f"Failed to download {ticker}: {exc}") from exc
    if frame.empty: raise RuntimeError(f"No market data returned for {ticker}.")
    if isinstance(frame.columns, pd.MultiIndex): frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    # yfinance normally calls the index Date; accept unnamed indexes as well.
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    frame["Date"] = pd.to_datetime(frame[date_column]).dt.normalize()
    frame["daily_return"] = frame["Close"].pct_change()
    # Negative shifts place *future* close returns on the current row; they are outcomes, never signal inputs.
    frame["next_day_return"] = frame["Close"].shift(-1) / frame["Close"] - 1
    frame["2_day_forward_return"] = frame["Close"].shift(-2) / frame["Close"] - 1
    frame["3_day_forward_return"] = frame["Close"].shift(-3) / frame["Close"] - 1
    return frame.rename(columns={"Date": "date"})


def align_sentiment_to_trading_days(sentiment: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Map a calendar-day sentiment observation to the next available trading day."""
    data = sentiment.copy(); dates = pd.DatetimeIndex(pd.to_datetime(market["date"]).dt.normalize().sort_values().unique())
    if dates.empty: raise ValueError("Market data has no trading dates.")
    data["sentiment_date"] = pd.to_datetime(data["date"]).dt.normalize()
    index = dates.searchsorted(data["sentiment_date"], side="left")
    data["date"] = [dates[i] if i < len(dates) else pd.NaT for i in index]
    return data.dropna(subset=["date"])
