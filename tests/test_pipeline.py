import pandas as pd
import pytest

from src.data_loader import DatasetValidationError, load_dataset
from src.market_data import fetch_market_data
from src.preprocessing import clean_text, normalize_label
from src.prepare_dataset import VERIFIED_LABEL_MAPPING, prepare_frame
from src.finbert import resolve_finbert_labels
from src.sentiment_index import daily_sentiment_index
from src.signals import create_signals
from src.train import predict_sentiment


def test_clean_text_preserves_financial_terms():
    assert clean_text("RT @user: $NVDA #bullish! https://x.com/a") == "$nvda bullish!"


@pytest.mark.parametrize("raw, expected", [(1, "positive"), (0, "neutral"), (-1, "negative"), ("BEARISH", "negative"), ("bad", None)])
def test_label_normalization(raw, expected): assert normalize_label(raw) == expected


def test_empty_prediction_rejected():
    with pytest.raises(ValueError, match="empty"): predict_sentiment("  ")


def test_sentiment_index():
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3), "ticker": ["AAPL"] * 3, "sentiment": ["positive", "neutral", "negative"]})
    result = daily_sentiment_index(frame).iloc[0]
    assert result.total_posts == 3 and result.sentiment_index == 0 and result.mean_sentiment == 0


def test_signal_thresholds():
    result = create_signals(pd.DataFrame({"sentiment_index": [.3, 0, -.3]}))
    assert result.tolist() == ["BUY", "HOLD", "SELL"]


def test_forward_return_direction(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame({"Open": [10, 20, 30], "High": [10, 20, 30], "Low": [10, 20, 30], "Close": [10, 20, 30], "Volume": [1, 1, 1]}, index=pd.date_range("2024-01-01", periods=3))
    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    result = fetch_market_data("AAPL", "2024-01-01", "2024-01-04")
    assert result.loc[0, "next_day_return"] == 1 and result.loc[0, "2_day_forward_return"] == 2


def test_malformed_dataset(tmp_path):
    path = tmp_path / "bad.csv"; path.write_text("message,label\nhello,positive\n")
    with pytest.raises(DatasetValidationError, match="Missing required"): load_dataset(path)


def test_huggingface_preparation_maps_labels_and_deduplicates():
    class Split:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def to_pandas(self): return pd.DataFrame(self.rows)
    source = {
        "train": Split([{"text": "Bearish outlook", "label": 0}, {"text": "", "label": 1}]),
        "validation": Split([{"text": "Bullish earnings", "label": 1}, {"text": "Bearish outlook", "label": 0}, {"text": "No change", "label": 2}]),
    }
    frame, sizes = prepare_frame(source)
    assert VERIFIED_LABEL_MAPPING == {0: "negative", 1: "positive", 2: "neutral"}
    assert sizes == {"train": 2, "validation": 3}
    assert frame.to_dict("records") == [{"text": "bearish outlook", "sentiment": "negative"}, {"text": "bullish earnings", "sentiment": "positive"}, {"text": "no change", "sentiment": "neutral"}]


def test_huggingface_preparation_rejects_unknown_label():
    class Split:
        def __len__(self): return 1
        def to_pandas(self): return pd.DataFrame([{"text": "Unknown", "label": 3}])
    with pytest.raises(ValueError, match="Unexpected source label"):
        prepare_frame({"train": Split(), "validation": Split()})


def test_finbert_label_mapping_is_semantic_not_position_based():
    labels = {0: "positive", 1: "negative", 2: "neutral"}
    assert resolve_finbert_labels(labels) == labels


def test_finbert_rejects_an_unexpected_label_configuration():
    with pytest.raises(RuntimeError, match="Unexpected ProsusAI/finbert label"):
        resolve_finbert_labels({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
