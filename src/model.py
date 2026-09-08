"""Weekly models and horizon-matched expanding-window evaluation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict
    comparison: pd.DataFrame
    folds: pd.DataFrame

class HoltWintersForecaster:
    def __init__(self, seasonal_periods=7, damped_trend=True, seed=42):
        self.period, self.damped, self.seed = seasonal_periods, damped_trend, seed
        self._fit = None

    def fit(self, series):
        y = series.y.to_numpy(dtype=float)
        if len(y) < 2*self.period or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Need two seasonal cycles of finite nonnegative demand.')
        dates = pd.DatetimeIndex(series.ds)
        if not dates.equals(pd.date_range(dates[0], periods=len(dates), freq='D')):
            raise ValueError('Dates must be ordered, unique, and continuous daily observations.')
        self._fit = ExponentialSmoothing(y, trend='add', damped_trend=self.damped,
            seasonal='add', seasonal_periods=self.period,
            initialization_method='estimated').fit(optimized=True)
        self.residuals = y-self._fit.fittedvalues
        return self

    def predict(self, last_date, horizon=30, interval=0.90):
        if self._fit is None:
            raise RuntimeError('Call fit before predict.')
        if horizon < 1 or not 0 < interval < 1:
            raise ValueError('Invalid horizon or interval level.')
        point = np.maximum(np.asarray(self._fit.forecast(horizon)), 0)
        # Approximation: does not propagate trend or parameter uncertainty.
        errors = self.residuals-self.residuals.mean()
        draws = point+np.random.default_rng(self.seed).choice(errors, (2000, horizon))
        low, high = np.quantile(draws, [(1-interval)/2, (1+interval)/2], axis=0)
        return pd.DataFrame({'ds': pd.date_range(pd.Timestamp(last_date)+pd.Timedelta(days=1), periods=horizon),
            'yhat': point, 'yhat_lower': np.maximum(low, 0), 'yhat_upper': np.maximum(high, 0)})

def seasonal_naive(values, horizon, period=7):
    """Repeat the last observed week without reading any holdout values."""
    if len(values) < period:
        raise ValueError('One complete week required.')
    return np.resize(np.asarray(values)[-period:], horizon)

def evaluate(series, horizon=30, windows=3):
    if horizon < 1 or windows < 1:
        raise ValueError('horizon and windows must be positive.')
    required = 14+horizon*windows
    if len(series) < required:
        raise ValueError(f'Need {required} daily observations for {windows} folds of {horizon} days.')
    rows = []
    for fold in range(windows):
        end = len(series)-horizon*(windows-fold)
        train, test = series.iloc[:end], series.iloc[end:end+horizon]
        pred = HoltWintersForecaster().fit(train).predict(train.ds.iloc[-1], horizon)
        actual = test.y.to_numpy()
        for name, point in [('Holt-Winters', pred.yhat.to_numpy()),
                            ('Seasonal naive', seasonal_naive(train.y, horizon))]:
            error = point-actual
            rows.append(dict(model=name, fold=fold+1, origin=train.ds.iloc[-1],
                test_end=test.ds.iloc[-1], horizon=horizon, n=len(test),
                absolute_error=np.abs(error).sum(), squared_error=np.square(error).sum(),
                actual_total=actual.sum(),
                coverage=np.mean((actual >= pred.yhat_lower) & (actual <= pred.yhat_upper))
                    if name == 'Holt-Winters' else np.nan))
    folds = pd.DataFrame(rows)
    summary = []
    for name, group in folds.groupby('model', sort=False):
        summary.append(dict(model=name, horizon=horizon, folds=windows,
            mae=group.absolute_error.sum()/group.n.sum(),
            rmse=np.sqrt(group.squared_error.sum()/group.n.sum()),
            wape=group.absolute_error.sum()/group.actual_total.sum() if group.actual_total.sum() else np.nan,
            coverage=group.coverage.mean()))
    return pd.DataFrame(summary), folds

def rolling_origin_backtest(series, horizon=30, windows=3):
    comparison, _ = evaluate(series, horizon, windows)
    return comparison.iloc[0][['mae','rmse','wape','coverage']].to_dict()

def train_and_forecast(series, horizon=30):
    comparison, folds = evaluate(series, horizon)
    forecast = HoltWintersForecaster().fit(series).predict(series.ds.iloc[-1], horizon)
    return ForecastResult(forecast, comparison.iloc[0][['mae','rmse','wape','coverage']].to_dict(), comparison, folds)
