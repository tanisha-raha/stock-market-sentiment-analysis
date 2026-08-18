"""Command line interface for stock-market sentiment analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import DatasetValidationError, load_dataset, require_market_columns
from src.evaluate import add_to_comparison, plot_model_comparison, plot_sentiment_distribution, save_comparison
from src.finbert import evaluate_finbert
from src.market_data import align_sentiment_to_trading_days, fetch_market_data
from src.sentiment_index import daily_sentiment_index
from src.signals import lag_analysis, signal_summary
from src.train import predict_sentiment, split_data, train_baselines


def market_plots(merged: pd.DataFrame, ticker: str) -> None:
    out = Path("outputs/figures"); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(merged.date, merged.sentiment_index); ax.axhline(0, color="black", lw=.8); ax.set(title=f"{ticker} daily sentiment", ylabel="Sentiment index"); fig.tight_layout(); fig.savefig(out / f"{ticker}_daily_sentiment.png", dpi=150); plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(10, 4)); ax1.plot(merged.date, merged.Close, color="tab:blue"); ax1.set_ylabel("Close"); ax2 = ax1.twinx(); ax2.plot(merged.date, merged.sentiment_index, color="tab:orange", alpha=.7); ax2.set_ylabel("Sentiment index"); ax1.set_title(f"{ticker}: price and sentiment"); fig.tight_layout(); fig.savefig(out / f"{ticker}_price_sentiment.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(merged.sentiment_index, merged.next_day_return, alpha=.7); ax.axhline(0, color="grey", lw=.8); ax.set(xlabel="Sentiment index", ylabel="Next-day return", title=f"{ticker}: sentiment vs next-day return"); fig.tight_layout(); fig.savefig(out / f"{ticker}_sentiment_scatter.png", dpi=150); plt.close(fig)


def run_market_analysis(frame: pd.DataFrame, ticker: str) -> None:
    require_market_columns(frame)
    subset = frame[frame.ticker == ticker.upper()].dropna(subset=["date"])
    if subset.empty: raise DatasetValidationError(f"No valid dated posts for ticker {ticker.upper()}.")
    sentiment = daily_sentiment_index(subset)
    market = fetch_market_data(ticker.upper(), sentiment.date.min().date(), (sentiment.date.max() + pd.Timedelta(days=7)).date())
    aligned = align_sentiment_to_trading_days(sentiment, market)
    merged = aligned.merge(market, on="date", how="inner")
    if merged.empty: raise DatasetValidationError("No sentiment rows align with returned market data.")
    lag = lag_analysis(merged); lag.to_csv("outputs/metrics/lag_analysis.csv", index=False)
    summary = signal_summary(merged)
    Path("outputs/metrics/signal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    market_plots(merged, ticker.upper())
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(lag.return_metric, lag.pearson_correlation, label="Pearson"); ax.bar(lag.return_metric, lag.spearman_correlation, alpha=.65, label="Spearman"); ax.axhline(0, color="black", lw=.8); ax.legend(); ax.set(title="Lag correlation comparison", ylabel="Correlation"); fig.tight_layout(); fig.savefig("outputs/figures/lag_correlation.png", dpi=150); plt.close(fig)
    print(lag.to_string(index=False)); print("\nExperimental behavioral signals:", json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial social-media sentiment analysis")
    parser.add_argument("--mode", choices=["train", "predict", "market-analysis", "finbert-evaluate"], required=True)
    parser.add_argument("--data", help="Local CSV path (required for train / market-analysis)")
    parser.add_argument("--text", help="Text to classify (required for predict)")
    parser.add_argument("--ticker", help="Ticker for market-analysis, e.g. NVDA")
    parser.add_argument("--text-column", default="text"); parser.add_argument("--sentiment-column", default="sentiment")
    parser.add_argument("--date-column", default="date"); parser.add_argument("--ticker-column", default="ticker")
    parser.add_argument("--finbert", action="store_true", help="Also evaluate FinBERT during training (internet required initially)")
    parser.add_argument("--finbert-batch-size", type=int, default=16, help="FinBERT inference batch size (default: 16)")
    args = parser.parse_args()
    try:
        if args.mode == "predict":
            if args.text is None: parser.error("--text is required for predict")
            print(json.dumps(predict_sentiment(args.text), indent=2)); return
        if not args.data: parser.error("--data is required for this mode")
        mapping = {"text": args.text_column, "sentiment": args.sentiment_column, "date": args.date_column, "ticker": args.ticker_column}
        frame, warnings = load_dataset(args.data, mapping)
        for warning in warnings: print("Warning:", warning)
        if args.mode == "train":
            plot_sentiment_distribution(frame, "outputs/figures")
            comparison, test_data = train_baselines(frame)
            if args.finbert:
                finbert = evaluate_finbert(*test_data); comparison = save_comparison([*comparison.to_dict("records"), finbert], "outputs/metrics/model_comparison.csv"); plot_model_comparison(comparison, "outputs/figures")
            print(comparison.to_string(index=False))
        elif args.mode == "finbert-evaluate":
            # Recreates train_baselines' deterministic held-out split without
            # fitting baseline models or changing their stored metrics.
            _, x_test, _, y_test = split_data(frame)
            finbert = evaluate_finbert(x_test, y_test, batch_size=args.finbert_batch_size)
            comparison = add_to_comparison(finbert, "outputs/metrics/model_comparison.csv")
            plot_model_comparison(comparison, "outputs/figures")
            print(pd.DataFrame([finbert]).drop(columns=["classification_report"]).to_string(index=False))
        else:
            if not args.ticker: parser.error("--ticker is required for market-analysis")
            run_market_analysis(frame, args.ticker)
    except (DatasetValidationError, ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__": main()
