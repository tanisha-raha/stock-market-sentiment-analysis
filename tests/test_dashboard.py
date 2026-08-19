from src.dashboard import aggregate_sentiment, behavioral_signal, extract_news_fields, normalize_ticker


def test_normalize_ticker():
    assert normalize_ticker(" nvda ") == "NVDA"
    assert normalize_ticker("BRK.B") == "BRK.B"
    assert normalize_ticker("NVDA; DROP") is None
    assert normalize_ticker("") is None


def test_aggregate_sentiment_and_signal():
    result = aggregate_sentiment(["positive", "positive", "neutral", "negative"])
    assert result["total"] == 4
    assert result["positive_percentage"] == 50
    assert result["sentiment_index"] == 0.25
    assert result["overall_sentiment"] == "Positive"
    assert behavioral_signal(result["sentiment_index"]) == "BUY"
    assert behavioral_signal(-0.25) == "SELL"
    assert behavioral_signal(0.0) == "HOLD"


def test_extract_news_fields_handles_new_yfinance_shape():
    item = {"content": {"title": "Nvidia beats estimates", "provider": {"displayName": "Example News"}, "pubDate": "2026-08-18T10:00:00Z", "canonicalUrl": {"url": "https://example.com/news"}}}
    assert extract_news_fields(item) == {"headline": "Nvidia beats estimates", "publisher": "Example News", "date": "2026-08-18T10:00:00Z", "link": "https://example.com/news"}


def test_extract_news_fields_handles_legacy_yfinance_shape():
    item = {"title": "Tesla falls", "publisher": "Example", "providerPublishTime": 0, "link": "https://example.com"}
    result = extract_news_fields(item)
    assert result["headline"] == "Tesla falls"
    assert result["publisher"] == "Example"
    assert result["date"] == "1970-01-01 00:00 UTC"
    assert result["link"] == "https://example.com"
