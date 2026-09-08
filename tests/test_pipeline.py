import pandas as pd
import pytest

from src.data_pipeline import prepare_daily_series


def test_daily_aggregation_fills_calendar_gaps_without_leakage():
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-03"], "sales": [2, 3, 4]})
    out = prepare_daily_series(raw, fill_missing_dates=True)
    assert out[["ds", "y"]].to_dict("records") == [
        {"ds": pd.Timestamp("2024-01-01"), "y": 5},
        {"ds": pd.Timestamp("2024-01-02"), "y": 0},
        {"ds": pd.Timestamp("2024-01-03"), "y": 4},
    ]
    assert pd.isna(out.loc[0, "rolling_7d_mean"])

def test_missing_days_rejected_by_default():
    with pytest.raises(ValueError, match='Missing calendar'):
        prepare_daily_series(pd.DataFrame({'date':['2024-01-01','2024-01-03'], 'sales':[1,2]}))

def test_intraday_records_aggregate():
    out = prepare_daily_series(pd.DataFrame({'date':['2024-01-01 09:00','2024-01-01 12:00'], 'sales':[1,2]}))
    assert out.y.tolist() == [3]
