"""CLI entry point for a reproducible baseline forecast."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline import load_sales_csv, prepare_daily_series
from model import train_and_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a retail demand forecast.")
    parser.add_argument("--input", default="data/retail_sales.csv")
    parser.add_argument("--output", default="outputs/forecast.csv")
    parser.add_argument("--horizon", type=int, default=30, choices=[30, 60, 90])
    args = parser.parse_args()
    series = prepare_daily_series(load_sales_csv(args.input))
    result = train_and_forecast(series, horizon=args.horizon)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.forecast.to_csv(args.output, index=False)
    print(f"Forecast written to {args.output}")
    print("Backtest metrics:", ", ".join(f"{k}={v:.3f}" for k, v in result.metrics.items()))


if __name__ == "__main__":
    main()

