import numpy as np
import pandas as pd
import pytest
from src.model import seasonal_naive, train_and_forecast

def test_naive_repeats_last_week():
    assert seasonal_naive(np.arange(14), 10).tolist() == [7,8,9,10,11,12,13,7,8,9]

@pytest.mark.parametrize('horizon', [30,60,90])
def test_horizon_and_temporal_boundaries(horizon):
    x = np.arange(400)
    series = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=400),
                           'y': 100+0.1*x+10*np.sin(2*np.pi*x/7)})
    result = train_and_forecast(series, horizon)
    assert len(result.forecast) == horizon
    assert result.forecast.ds.iloc[0] == series.ds.iloc[-1]+pd.Timedelta(days=1)
    assert result.folds.horizon.eq(horizon).all()
    assert ((result.folds.test_end-result.folds.origin).dt.days == horizon).all()
    assert result.comparison.model.tolist() == ['Holt-Winters','Seasonal naive']
    assert result.forecast.yhat_lower.le(result.forecast.yhat).all()
    assert result.forecast.yhat.le(result.forecast.yhat_upper).all()

