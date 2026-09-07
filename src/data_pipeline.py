"""Data loading, validation, and leakage-safe daily series preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"date", "sales"}


def load_sales_csv(path: str | Path) -> pd.DataFrame:
    """Load a retail CSV and normalize its date/sales fields."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    if (df["sales"] < 0).any():
        raise ValueError("Sales cannot be negative after cleaning.")
    return df


def prepare_daily_series(
    df: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    fill_missing_dates: bool = True,
) -> pd.DataFrame:
    """Aggregate transactions to a continuous daily ``ds, y`` series.

    Missing calendar dates are filled with zero only after aggregation. This
    makes the assumption explicit: a missing date means no recorded demand.
    If missingness means data collection failure in the source system, callers
    should pass ``fill_missing_dates=False`` and investigate the gaps.
    """
    required = {"date", "sales"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {sorted(required)}")
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["sales"] = pd.to_numeric(work["sales"], errors="coerce")
    work = work.dropna(subset=["date", "sales"])
    work = work.loc[work["sales"] >= 0]

    daily = work.groupby("date", as_index=False)["sales"].sum()
    daily = daily.rename(columns={"date": "ds", "sales": "y"}).sort_values("ds")
    if daily.empty:
        raise ValueError("No valid observations remain after cleaning.")

    if start or end:
        start_ts = pd.Timestamp(start) if start else daily["ds"].min()
        end_ts = pd.Timestamp(end) if end else daily["ds"].max()
        daily = daily.loc[daily["ds"].between(start_ts, end_ts)]
    if daily.empty:
        raise ValueError("The requested date window contains no observations.")

    if fill_missing_dates:
        index = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        daily = daily.set_index("ds").reindex(index, fill_value=0.0)
        daily.index.name = "ds"
        daily = daily.reset_index()
    return add_calendar_features(daily)


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features for diagnostics and future model extensions."""
    result = daily.copy()
    result["day_of_week"] = result["ds"].dt.dayofweek
    result["week_of_year"] = result["ds"].dt.isocalendar().week.astype(int)
    result["month"] = result["ds"].dt.month
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["rolling_7d_mean"] = result["y"].shift(1).rolling(7, min_periods=1).mean()
    result["rolling_28d_mean"] = result["y"].shift(1).rolling(28, min_periods=1).mean()
    return result

