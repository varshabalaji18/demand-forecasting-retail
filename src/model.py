"""Forecast model and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:  # Allows the data pipeline and demo CLI to import before setup.
    ExponentialSmoothing = None


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict[str, float]


class _FallbackFit:
    """Small dependency-free seasonal trend fallback for smoke tests."""

    def __init__(self, values: np.ndarray, period: int):
        self.values = values
        self.period = period
        x = np.arange(len(values))
        self.slope, self.intercept = np.polyfit(x, values, 1)
        detrended = values - (self.intercept + self.slope * x)
        self.seasonal = np.array([detrended[-period + i :: period].mean() for i in range(period)])
        self.fittedvalues = self.intercept + self.slope * x + np.resize(self.seasonal, len(values))

    def forecast(self, horizon: int) -> np.ndarray:
        x = np.arange(len(self.values), len(self.values) + horizon)
        return self.intercept + self.slope * x + np.resize(self.seasonal, horizon)


class HoltWintersForecaster:
    """Daily demand forecast using trend + weekly seasonality.

    Intervals are residual-bootstrap predictive intervals, not parameter
    confidence intervals. This is a practical planning interval for a demo
    product and is clearly labeled in the app/README.
    """

    def __init__(self, seasonal_periods: int = 7, damped_trend: bool = True, seed: int = 42):
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.rng = np.random.default_rng(seed)
        self._fit = None
        self._residuals = None

    def fit(self, series: pd.DataFrame) -> "HoltWintersForecaster":
        if not {"ds", "y"}.issubset(series.columns):
            raise ValueError("Series must contain ds and y columns.")
        values = series["y"].astype(float).to_numpy()
        if len(values) < self.seasonal_periods * 2:
            raise ValueError("At least two seasonal cycles are required for training.")
        if ExponentialSmoothing is None:
            self._fit = _FallbackFit(values, self.seasonal_periods)
        else:
            self._fit = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=self.damped_trend,
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=True)
        self._residuals = values - self._fit.fittedvalues
        return self

    def predict(self, last_date: pd.Timestamp, horizon: int = 30, interval: float = 0.90) -> pd.DataFrame:
        if self._fit is None:
            raise RuntimeError("Call fit() before predict().")
        if horizon < 1 or not 0 < interval < 1:
            raise ValueError("horizon must be positive and interval must be between 0 and 1.")
        point = np.asarray(self._fit.forecast(horizon), dtype=float)
        residuals = self._residuals[np.isfinite(self._residuals)]
        draws = point[None, :] + self.rng.choice(residuals, size=(2000, horizon), replace=True)
        alpha = 1 - interval
        low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
        dates = pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.DataFrame({"ds": dates, "yhat": np.maximum(point, 0), "yhat_lower": np.maximum(low, 0), "yhat_upper": np.maximum(high, 0)})


def backtest(series: pd.DataFrame, horizon: int = 30) -> dict[str, float]:
    """Evaluate the final rolling-origin holdout without leaking future data."""
    if len(series) <= horizon + 14:
        raise ValueError("Not enough observations for the requested holdout.")
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    model = HoltWintersForecaster().fit(train)
    pred = model.predict(train["ds"].iloc[-1], horizon=horizon)
    actual, forecast = test["y"].to_numpy(), pred["yhat"].to_numpy()
    errors = actual - forecast
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "wape": float(np.abs(actual - forecast).sum() / max(np.abs(actual).sum(), 1e-9)),
    }


def rolling_origin_backtest(series: pd.DataFrame, horizon: int = 30, windows: int = 3) -> dict[str, float]:
    """Evaluate several expanding-window forecast origins.

    Every fold trains only on observations available at that origin. This is
    closer to deployment behavior than a random split and reduces dependence
    on one unusually easy or difficult final holdout.
    """
    if horizon < 1 or windows < 1:
        raise ValueError("horizon and windows must be positive.")
    minimum_train = 14
    required = minimum_train + horizon * windows
    if len(series) < required:
        return backtest(series, horizon=min(horizon, max(7, len(series) // 5)))

    fold_metrics = []
    for fold in range(windows, 0, -1):
        train_end = len(series) - horizon * fold
        train, test = series.iloc[:train_end], series.iloc[train_end : train_end + horizon]
        model = HoltWintersForecaster().fit(train)
        pred = model.predict(train["ds"].iloc[-1], horizon=horizon)
        actual, forecast = test["y"].to_numpy(), pred["yhat"].to_numpy()
        errors = actual - forecast
        fold_metrics.append({
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "wape": float(np.abs(errors).sum() / max(np.abs(actual).sum(), 1e-9)),
        })
    return {metric: float(np.mean([fold[metric] for fold in fold_metrics])) for metric in fold_metrics[0]}


def train_and_forecast(series: pd.DataFrame, horizon: int = 30) -> ForecastResult:
    metrics = rolling_origin_backtest(series, horizon=min(horizon, 30), windows=3)
    model = HoltWintersForecaster().fit(series)
    forecast = model.predict(series["ds"].iloc[-1], horizon=horizon)
    return ForecastResult(forecast=forecast, metrics=metrics)
