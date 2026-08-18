# Stock Market Sentiment Analysis using X/Twitter NLP

A reproducible portfolio project that classifies labelled financial social-media posts as positive, neutral, or negative, then investigates whether daily aggregate sentiment is associated with short-term stock returns. It is deliberately an analysis tool, not a trading system.

## Research question

Can public sentiment about major stocks help explain or anticipate short-term stock-price movements? The pipeline supports AAPL, TSLA, NVDA, MSFT, AMZN, and other ticker symbols supplied in the dataset.

## Pipeline

`Local historical CSV → validation/cleaning → train/test split → TF-IDF → Naive Bayes + Logistic Regression (+ optional FinBERT) → daily sentiment index → trading-day alignment → return/correlation/signal analysis`

## Features

- Configurable input names standardised to `text`, `sentiment`, `date`, and `ticker`.
- Robust validation for missing fields, invalid labels/dates, blank posts, and duplicates.
- Leakage-safe TF-IDF baselines: the vectorizer is fitted only on training text.
- Saved vectorizer and models, plus reusable prediction.
- Optional batched `ProsusAI/finbert` inference, using CUDA, Apple MPS, or CPU when available.
- Daily/engagement-weighted sentiment, calendar-to-next-trading-day alignment, forward returns, Pearson/Spearman lags, and experimental behavioural signals.

## Repository structure

```text
data/raw/                 # Put your CSV here (not committed)
data/processed/           # Optional derived data
notebooks/                # Guided EDA and analysis notebooks
src/                      # Reusable pipeline modules
models/                   # Generated .joblib model artifacts
outputs/figures/          # Generated plots
outputs/metrics/          # Generated CSV/JSON results
tests/                    # Offline pytest suite
main.py                   # CLI entry point
```

## Installation

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset requirements

The primary labelled dataset for the sentiment-classification experiment is
[`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment).
Prepare it with:

```bash
python -m src.prepare_dataset
```

This downloads its `train` and `validation` splits, verifies the documented mapping
`0 = Bearish`, `1 = Bullish`, `2 = Neutral`, maps them to `negative`, `positive`,
and `neutral`, cleans/removes duplicate text, and writes `data/raw/dataset.csv`.
The combined source corpus is then subject to this project's own stratified 80/20
holdout split, before TF-IDF fitting, for a reproducible no-leakage evaluation.

This dataset is for sentiment-model training/evaluation only. It does not provide
the reliable date and stock-ticker fields needed for the market-analysis stage.
Market analysis and a future live dashboard should use a separate, recent,
stock-specific data source.

The CSV must have labelled text columns. The default required columns are:

| Column | Required for | Accepted values |
|---|---|---|
| `text` | classification | post text |
| `sentiment` | classification | positive/neutral/negative, or 1/0/-1 |
| `date` | market analysis | parseable date |
| `ticker` | market analysis | e.g. NVDA |

`date` and `ticker` are optional for classification. Use flags such as `--text-column tweet --sentiment-column label` for a differently named CSV. Optional `likes`/`favorites` and `retweets`/`reposts` enable conservative engagement weighting: `1 + log1p(likes + 2*retweets)`.

Place the file at `data/raw/dataset.csv` (or pass any local path). The repository never invents a dataset or results.

## Usage

```bash
# Train and save baselines; produces actual metrics and confusion matrices
python main.py --mode train --data data/raw/dataset.csv

# Evaluate pretrained FinBERT on the same deterministic held-out split
# (requires initial model download; does not retrain baseline models)
python main.py --mode finbert-evaluate --data data/raw/dataset.csv

# Predict after training
python main.py --mode predict --text "Nvidia smashed earnings expectations"

# Analyse one ticker (requires date + ticker and internet for yfinance)
python main.py --mode market-analysis --data data/raw/dataset.csv --ticker NVDA
```

## NLP methodology and evaluation

Cleaning removes URLs, mentions, RT markers, HTML artifacts, and hashtag symbols while retaining words such as *bullish*, *earnings*, *buy*, and *crash*. An approximately 80/20 stratified split uses `random_state=42`; TF-IDF uses word unigrams/bigrams and is fitted after splitting. The project evaluates Naive Bayes, Logistic Regression, and optionally FinBERT with accuracy, macro/weighted F1, macro precision/recall, reports, and confusion matrices. Results are written only after you run an experiment: **Run the experiment to populate this result.**

## Sentiment and market analysis

Daily index: `(positive_count - negative_count) / total_posts`, alongside mean score (+1/0/-1), class counts, and proportions. Sentiment on a weekend/holiday is mapped to the next available trading session so it is not silently discarded. Without timestamps, pre-market, intraday, and after-hours posts cannot be separated.

Market data comes from yfinance. The analysis compares same-day, next-day, 2-day, and 3-day forward returns. Forward returns are outcome columns only; they are never used to make historical signals. Correlation describes association, not causation.

Signals are explicitly **experimental behavioural signals**: BUY at index ≥ 0.25, SELL at ≤ -0.25, otherwise HOLD. The command reports counts and directional hit rates; it does not claim profitability.

## Visualizations

Generated outputs include sentiment distribution/model comparison/confusion matrices (from training) and daily sentiment, price with sentiment, next-day scatter, and lag correlations (from market analysis).

## Notebooks

The four notebooks are lightweight, guided starting points: EDA, preprocessing, baselines, and market analysis. Run cells after adding your own data; they do not contain invented findings.

## Limitations and disclaimer

Social-media samples can have selection bias, bots/spam, sarcasm, context ambiguity, and historical coverage limitations. Markets also reflect news and many confounders. X API access is not required and this project does not scrape X; yfinance/FinBERT downloads need internet. Transaction costs, slippage, and rigorous backtesting are not modeled. This is educational research, not financial advice; correlation does not establish causation and behavioural signals are experimental.

## Future improvements

Use timestamp-aware alignment, a documented data collection process, bot filtering, temporal validation, FinBERT fine-tuning where justified, and a cost-aware out-of-sample backtest.
