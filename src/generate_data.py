"""Generate a realistic, reproducible retail panel for local development."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_retail_data(output: str | Path = "data/retail_sales.csv", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    rows = []
    for store in range(1, 6):
        for item in range(1, 4):
            base = 55 + store * 7 + item * 10
            for date in dates:
                weekly = [0.78, 0.88, 0.96, 1.02, 1.15, 1.32, 1.22][date.dayofweek]
                yearly = 1 + 0.12 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                trend = 1 + 0.00035 * (date - dates[0]).days
                promo = 1.18 if date.day in (1, 15) else 1.0
                expected = base * weekly * yearly * trend * promo
                rows.append((date, store, item, max(0, rng.poisson(expected))))
    result = pd.DataFrame(rows, columns=["date", "store", "item", "sales"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    generate_retail_data()

