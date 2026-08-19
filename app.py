"""Interactive live financial-news sentiment dashboard. Run: streamlit run app.py"""
from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

from src.dashboard import (aggregate_sentiment, behavioral_signal, classify_headlines,
                           extract_news_fields, load_logistic_artifacts, normalize_ticker)

st.set_page_config(page_title="Stock Market Sentiment Analyzer", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
  .block-container { max-width: 1180px; padding-top: 3.1rem; padding-bottom: 2.75rem; }
  h1 { font-size: 2.25rem !important; font-weight: 650 !important; letter-spacing: -0.045em; line-height: 1.13; margin-bottom: .18rem !important; }
  h2 { font-size: 1.08rem !important; font-weight: 650 !important; letter-spacing: -0.018em; margin-top: 1.85rem !important; margin-bottom: .7rem !important; }
  [data-testid="stSidebar"] { background: #10161f; border-right: 1px solid #263140; }
  [data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
  [data-testid="stSidebar"] h3 { font-size: .78rem; letter-spacing: .03em; text-transform: uppercase; color: #94a3b8; }
  div[data-testid="stButton"] button { min-height: 38px; background: #38465d; border: 1px solid #52627b; border-radius: 7px; color: #f8fafc; font-size: .87rem; font-weight: 600; letter-spacing: .005em; }
  div[data-testid="stButton"] button:hover { background: #465670; border-color: #64748b; color: #fff; }
  div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background: #121b28; border-color: #344154; border-radius: 7px; min-height: 38px; }
  div[data-testid="stTextInput"] label, div[data-testid="stSelectbox"] label { color: #94a3b8; font-size: .76rem; font-weight: 600; }
  .subtitle, .muted { color: #8f9bac; font-size: .84rem; }
  .toolbar-spacer { height: 1.35rem; }
  .empty-state { color: #8f9bac; font-size: .9rem; padding: 2.25rem 0 0; border-top: 1px solid #263140; margin-top: 1.5rem; }
  .empty-state strong { color: #d7dee8; font-weight: 550; }
  .metric-card { border: 1px solid #2b3748; border-radius: 11px; background: #131d2a; padding: .9rem 1rem; min-height: 94px; }
  .metric-label { color: #8f9bac; font-size: .71rem; font-weight: 650; letter-spacing: .045em; text-transform: uppercase; }
  .metric-value { color: #eef2f7; font-size: 1.27rem; font-weight: 620; line-height: 1.3; margin-top: .36rem; }
  .metric-detail { color: #78869a; font-size: .75rem; margin-top: .18rem; }
  .pos { color: #6fca9c; } .neg { color: #e49393; } .neu { color: #c4ceda; }
  .sentiment-stack { display: flex; overflow: hidden; height: 9px; background: #263140; border-radius: 99px; margin: .9rem 0 .65rem; }
  .sentiment-stack > span { height: 100%; } .positive { background: #4fae84; } .neutral { background: #8d9bac; } .negative { background: #c67878; }
  .sentiment-legend { display: flex; justify-content: space-between; gap: .5rem; color: #9ba8b9; font-size: .76rem; }
  .signal-card { border: 1px solid #2f3d50; border-left: 2px solid #63738d; border-radius: 8px; padding: .85rem .95rem; margin-top: 1.1rem; background: rgba(19,29,42,.55); }
  .signal-value { font-size: 1.12rem; font-weight: 700; letter-spacing: .04em; }
  .headline-card { border-top: 1px solid #273449; padding: .85rem 0; }
  .headline-card:first-child { border-top: 0; padding-top: .1rem; }
  .headline-title { color: #e2e8f0; font-size: .94rem; font-weight: 600; line-height: 1.35; }
  .headline-meta { color: #94a3b8; font-size: .78rem; margin-top: .25rem; }
  .badge { float: right; border-radius: 5px; font-size: .67rem; font-weight: 650; letter-spacing: .025em; padding: .16rem .42rem; }
  .badge-positive { background: rgba(79,174,132,.13); color: #78c99f; } .badge-neutral { background: rgba(141,155,172,.13); color: #c4ceda; } .badge-negative { background: rgba(198,120,120,.13); color: #e49393; }
  .source-link { color: #9fb9d9; font-size: .78rem; text-decoration: none; }
  .source-link:hover { color: #c7d9ee; text-decoration: underline; }
  @media (max-width: 700px) { .block-container { padding: 1.7rem 1rem 2rem; } h1 { font-size: 1.9rem !important; } .sentiment-legend { flex-wrap: wrap; } }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_logistic_artifacts(Path(__file__).parent / "models")


@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker: str, period: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)


@st.cache_data(ttl=900, show_spinner=False)
def get_snapshot(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    try: info = stock.info or {}
    except Exception: info = {}
    history = stock.history(period="5d", interval="1d", auto_adjust=False)
    if history.empty or "Close" not in history:
        raise ValueError("No recent price data was returned for this ticker.")
    closes = history["Close"].dropna()
    latest = float(closes.iloc[-1])
    previous = info.get("previousClose") or info.get("regularMarketPreviousClose")
    if previous is None and len(closes) > 1: previous = float(closes.iloc[-2])
    change = ((latest - float(previous)) / float(previous) * 100) if previous else None
    return {"company_name": info.get("longName") or info.get("shortName") or ticker, "price": latest,
            "previous_close": previous, "daily_change": change, "volume": int(history["Volume"].fillna(0).iloc[-1]) if "Volume" in history else None,
            "market_cap": info.get("marketCap"), "week_high": info.get("fiftyTwoWeekHigh"), "week_low": info.get("fiftyTwoWeekLow")}


@st.cache_data(ttl=900, show_spinner=False)
def get_news(ticker: str) -> list[dict]:
    try: raw = yf.Ticker(ticker).news or []
    except Exception as exc: raise RuntimeError(f"Could not retrieve recent news: {exc}") from exc
    records = [extract_news_fields(item) for item in raw if isinstance(item, dict)]
    return [record for record in records if record["headline"]][:10]


def money(value) -> str:
    if value is None: return "—"
    if abs(float(value)) >= 1_000_000_000: return f"${float(value) / 1_000_000_000:.2f}B"
    if abs(float(value)) >= 1_000_000: return f"${float(value) / 1_000_000:.2f}M"
    return f"${float(value):,.2f}"


def metric_card(label: str, value: str, detail: str = "", accent: str = "") -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value {accent}">{escape(value)}</div><div class="metric-detail">{escape(detail)}</div></div>', unsafe_allow_html=True)


def safe_link(value: str | None) -> str | None:
    if not value: return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def sentiment_bars(summary: dict) -> None:
    positive, neutral, negative = (float(summary[f"{label}_percentage"]) for label in ("positive", "neutral", "negative"))
    st.markdown(f'''<div class="sentiment-stack"><span class="positive" style="width:{positive:.1f}%"></span><span class="neutral" style="width:{neutral:.1f}%"></span><span class="negative" style="width:{negative:.1f}%"></span></div><div class="sentiment-legend"><span>Positive {positive:.0f}%</span><span>Neutral {neutral:.0f}%</span><span>Negative {negative:.0f}%</span></div>''', unsafe_allow_html=True)


def headline_card(row: dict) -> None:
    sentiment = row.get("sentiment", "neutral")
    source = safe_link(row.get("link"))
    source_html = f' · <a class="source-link" href="{escape(source, quote=True)}" target="_blank" rel="noopener noreferrer">Open source →</a>' if source else ""
    title = escape(str(row.get("headline") or "Untitled headline"))
    publisher = escape(str(row.get("publisher") or "Unknown publisher"))
    date = escape(str(row.get("date") or "Date unavailable"))
    st.markdown(f'<div class="headline-card"><span class="badge badge-{escape(sentiment)}">{escape(sentiment.title())}</span><div class="headline-title">{title}</div><div class="headline-meta">{publisher} · {date}{source_html}</div></div>', unsafe_allow_html=True)


def render_dashboard(ticker: str, period: str, period_label: str) -> None:
    try:
        snapshot, history = get_snapshot(ticker), get_history(ticker, period)
    except Exception as exc:
        st.error(f"Unable to retrieve market data for {ticker}: {exc}")
        return
    if history.empty:
        st.error(f"No price history is currently available for {ticker}.")
        return
    try:
        model, vectorizer = get_model()
        model_error = None
    except Exception as exc:
        model, vectorizer, model_error = None, None, exc
    news: list[dict] = []
    try: news = get_news(ticker)
    except Exception as exc: st.warning(str(exc))
    classifications = classify_headlines([row["headline"] for row in news], model, vectorizer) if model else []
    for row, prediction in zip(news, classifications): row.update(prediction)
    summary = aggregate_sentiment(row["sentiment"] for row in news if "sentiment" in row)

    st.markdown(f'<div style="margin-top:1.8rem"><span style="font-size:1.4rem;font-weight:650;letter-spacing:-.025em">{escape(ticker)}</span><span class="muted" style="margin-left:.55rem">{escape(str(snapshot["company_name"]))}</span></div>', unsafe_allow_html=True)
    cards = st.columns(4, gap="small")
    with cards[0]: metric_card("Current Price", money(snapshot["price"]), "Latest available close")
    change = snapshot["daily_change"]
    with cards[1]: metric_card("Daily Change", "—" if change is None else f"{change:+.2f}%", "vs. previous close", "pos" if change and change > 0 else "neg" if change and change < 0 else "neu")
    overall = summary["overall_sentiment"] if summary["total"] else "Unavailable"
    with cards[2]: metric_card("Sentiment", overall, "Recent financial headlines", "pos" if "Positive" in overall else "neg" if "Negative" in overall else "neu")
    with cards[3]: metric_card("Sentiment Score", f"{summary['sentiment_index']:+.2f}" if summary["total"] else "—", "Range: −1 to +1")

    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.header("Price History")
        prices = history.reset_index()
        date_column = "Date" if "Date" in prices else prices.columns[0]
        chart = alt.Chart(prices).mark_line(color="#60a5fa", strokeWidth=2.3).encode(
            x=alt.X(f"{date_column}:T", title=None, axis=alt.Axis(format="%b %d", grid=False, labelColor="#94a3b8")),
            y=alt.Y("Close:Q", title="Close", axis=alt.Axis(gridColor="#263449", labelColor="#94a3b8")),
            tooltip=[alt.Tooltip(f"{date_column}:T", title="Date"), alt.Tooltip("Close:Q", title="Close", format="$.2f")]
        ).properties(height=350, title=alt.TitleParams(f"Closing price · {period_label}", color="#cbd5e1", fontSize=13, anchor="start"))
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.header("Sentiment Overview")
        with st.container(border=True):
            if model_error:
                st.caption("Headline classification unavailable")
                st.write("Saved Logistic Regression artifacts are missing.")
            elif not news:
                st.caption("No recent headlines are currently available.")
            else:
                st.markdown(f'**{summary["overall_sentiment"]}** <span class="muted">· {summary["total"]} headlines</span>', unsafe_allow_html=True)
                sentiment_bars(summary)
                st.caption(f"Overall sentiment: {summary['overall_sentiment']} · Score: {summary['sentiment_index']:+.2f}")
                signal = behavioral_signal(float(summary["sentiment_index"]))
                st.markdown(f'<div class="signal-card"><div class="eyebrow">Experimental Behavioral Signal</div><div class="signal-value">{signal}</div><div class="muted">Based on recent financial-news sentiment only.</div><div class="muted" style="margin-top:.35rem">Educational sentiment indicator only. Not financial advice.</div></div>', unsafe_allow_html=True)

    st.header("Market Overview")
    overview = st.columns(4, gap="small")
    with overview[0]: metric_card("Volume", f"{snapshot['volume']:,}" if snapshot["volume"] is not None else "—")
    with overview[1]: metric_card("Market Cap", money(snapshot["market_cap"]))
    with overview[2]: metric_card("52W High", money(snapshot["week_high"]))
    with overview[3]: metric_card("52W Low", money(snapshot["week_low"]))

    st.header("Recent Headlines")
    if model_error:
        st.info(f"Saved Logistic Regression model is unavailable, so headline sentiment was not calculated. {model_error}")
    elif not news:
        st.info("No recent financial headlines are currently available, so sentiment cannot be calculated.")
    else:
        with st.expander(f"Recent Financial News ({len(news)})", expanded=True):
            for row in news: headline_card(row)


st.title("Stock Market Sentiment Analyzer")
st.markdown('<div class="subtitle">Live market data and financial-news sentiment, analyzed with machine learning.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Analysis Settings")
    st.caption("Search controls are available in the main header.")
    st.markdown("### Model")
    st.write("Logistic Regression + TF-IDF")
    st.markdown("### Data Sources")
    st.caption("Yahoo Finance via yfinance\n\nTwitter Financial News Sentiment dataset")
    with st.expander("About & limitations"):
        st.caption("Headlines are incomplete and can be ambiguous or delayed. Market movements have many confounding factors.")
    with st.expander("Disclaimer"):
        st.caption("Not affiliated with Yahoo Finance, X, or any financial institution. Not financial advice.")

st.markdown('<div class="toolbar-spacer"></div>', unsafe_allow_html=True)
input_column, period_column, button_column = st.columns([48, 30, 22], gap="small")
with input_column: entered = st.text_input("Enter stock ticker", value="NVDA", label_visibility="visible")
with period_column: period_label = st.selectbox("Time range", ["1 Month", "3 Months", "6 Months", "1 Year"], index=1)
period = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}[period_label]
with button_column:
    st.write("")
    analyze = st.button("Analyze", use_container_width=True)

if analyze:
    ticker = normalize_ticker(entered)
    if not ticker: st.error("Enter a valid ticker, for example NVDA, AAPL, TSLA, MSFT, or AMZN.")
    else:
        with st.spinner(f"Retrieving current data for {ticker}..."): render_dashboard(ticker, period, period_label)
else:
    st.markdown('<div class="empty-state"><strong>Search a ticker to begin</strong><br><span>NVDA · AAPL · TSLA · MSFT · AMZN</span></div>', unsafe_allow_html=True)

with st.expander("About the Model"):
    st.caption("Logistic Regression + TF-IDF is the primary deployed baseline. It achieved approximately 79.2% accuracy and 0.734 macro F1 on the labelled financial Twitter/news experiment. Naive Bayes and FinBERT were evaluated as comparison models. Metrics are dataset-specific and do not guarantee performance on current market news.")
