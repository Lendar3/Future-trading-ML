from __future__ import annotations

import pandas as pd


def build_sentiment_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in (
        "news_count_1h",
        "news_count_24h",
        "sentiment_score",
        "sentiment_zscore",
        "negative_news_burst",
        "positive_news_burst",
        "regulatory_news_flag",
        "hack_news_flag",
        "etf_news_flag",
        "macro_news_flag",
        "social_mention_burst",
    ):
        out[f"feature_{name}"] = frame.get(name, 0.0)
    return out
