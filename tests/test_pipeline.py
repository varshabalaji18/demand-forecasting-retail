import pandas as pd

from src.data_pipeline import prepare_daily_series


def test_daily_aggregation_fills_calendar_gaps_without_leakage():
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-03"], "sales": [2, 3, 4]})
    out = prepare_daily_series(raw)
    assert out[["ds", "y"]].to_dict("records") == [
        {"ds": pd.Timestamp("2024-01-01"), "y": 5},
        {"ds": pd.Timestamp("2024-01-02"), "y": 0},
        {"ds": pd.Timestamp("2024-01-03"), "y": 4},
    ]
    assert pd.isna(out.loc[0, "rolling_7d_mean"])

