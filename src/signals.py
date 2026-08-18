"""Association and experimental behavioural-signal analysis."""
from __future__ import annotations

import pandas as pd


def create_signals(frame: pd.DataFrame, buy_threshold: float = 0.25, sell_threshold: float = -0.25, column: str = "sentiment_index") -> pd.Series:
    if buy_threshold <= sell_threshold: raise ValueError("buy_threshold must exceed sell_threshold.")
    return pd.Series("HOLD", index=frame.index).mask(frame[column] >= buy_threshold, "BUY").mask(frame[column] <= sell_threshold, "SELL")


def lag_analysis(merged: pd.DataFrame, sentiment_column: str = "sentiment_index") -> pd.DataFrame:
    rows = []
    for target in ("daily_return", "next_day_return", "2_day_forward_return", "3_day_forward_return"):
        subset = merged[[sentiment_column, target]].dropna()
        rows.append({"return_metric": target, "n": len(subset), "pearson_correlation": subset.corr(method="pearson").iloc[0, 1] if len(subset) >= 2 else float("nan"), "spearman_correlation": subset.corr(method="spearman").iloc[0, 1] if len(subset) >= 2 else float("nan")})
    return pd.DataFrame(rows)


def signal_summary(merged: pd.DataFrame, buy_threshold: float = .25, sell_threshold: float = -.25) -> dict:
    data = merged.copy(); data["signal"] = create_signals(data, buy_threshold, sell_threshold)
    data["next_day_direction"] = (data["next_day_return"] > 0).astype("Int64")
    buy = data[data.signal == "BUY"].dropna(subset=["next_day_return"]); sell = data[data.signal == "SELL"].dropna(subset=["next_day_return"])
    acted = data[data.signal != "HOLD"].dropna(subset=["next_day_return"])
    return {"buy_signals": len(buy), "sell_signals": len(sell), "buy_directional_hit_rate": (buy.next_day_return > 0).mean() if len(buy) else None, "sell_directional_hit_rate": (sell.next_day_return <= 0).mean() if len(sell) else None, "overall_directional_accuracy": (((acted.signal == "BUY") == (acted.next_day_return > 0)).mean() if len(acted) else None)}

