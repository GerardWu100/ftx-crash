"""Convert the original project inputs into filtered parquet files.

This script records the one-off conversion used to build the parquet runtime
inputs under ``data/processed/``. It expects the notebook-source files to be
restored under ``data/raw/`` before execution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

BTC_COLUMNS = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
BITO_COLUMNS = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
BTC_START = pd.Timestamp("2022-04-01")
BTC_END = pd.Timestamp("2022-12-31 23:59:59")
BITO_START = pd.Timestamp("2022-10-01")
BITO_END = pd.Timestamp("2022-12-31 23:59:59")
PERP_START = pd.Timestamp("2022-01-01")
PERP_END = pd.Timestamp("2023-01-01")

QUARTERLY_FILENAMES = (
    "DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2022-03_2022-06_binance_quarterly.csv",
    "DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2022-06_2022-09_binance_quarterly.csv",
)


def write_btc_spot_parquet() -> None:
    """Convert the BTC minute text file into the project runtime parquet file."""
    source_path = RAW_DATA_DIR / "BTC_full_1min.txt"
    target_path = PROCESSED_DATA_DIR / "BTC_full_1min.parquet"

    data = pd.read_csv(source_path, names=BTC_COLUMNS, parse_dates=["DateTime"])
    data = data.loc[
        (data["DateTime"] >= BTC_START) & (data["DateTime"] <= BTC_END)
    ].copy()
    data.to_parquet(target_path, index=False)
    print(f"Wrote {target_path.name}: {len(data):,} rows")


def write_bito_parquet() -> None:
    """Convert the BITO minute text file into the project runtime parquet file."""
    source_path = RAW_DATA_DIR / "BITO_full_1min_adjsplitdiv.txt"
    target_path = PROCESSED_DATA_DIR / "BITO_full_1min_adjsplitdiv.parquet"

    data = pd.read_csv(source_path, names=BITO_COLUMNS, parse_dates=["DateTime"])
    data = data.loc[
        (data["DateTime"] >= BITO_START) & (data["DateTime"] <= BITO_END)
    ].copy()
    data.to_parquet(target_path, index=False)
    print(f"Wrote {target_path.name}: {len(data):,} rows")


def write_perpetual_parquet() -> None:
    """Convert the BTC perpetual futures CSV into the runtime parquet file."""
    source_path = (
        RAW_DATA_DIR
        / "DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2019-2025_binance_perpetual.csv"
    )
    target_path = (
        PROCESSED_DATA_DIR
        / "DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2019-2025_binance_perpetual.parquet"
    )

    data = pd.read_csv(source_path)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.loc[
        (data["datetime"] >= PERP_START) & (data["datetime"] <= PERP_END)
    ].copy()
    data.to_parquet(target_path, index=False)
    print(f"Wrote {target_path.name}: {len(data):,} rows")


def write_quarterly_parquet(filename: str) -> None:
    """Convert one quarterly futures CSV into a runtime parquet file."""
    source_path = RAW_DATA_DIR / filename
    target_path = PROCESSED_DATA_DIR / f"{source_path.stem}.parquet"

    data = pd.read_csv(source_path)
    data.to_parquet(target_path, index=False)
    print(f"Wrote {target_path.name}: {len(data):,} rows")


REQUIRED_RAW_FILENAMES = (
    "BTC_full_1min.txt",
    "BITO_full_1min_adjsplitdiv.txt",
    "DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2019-2025_binance_perpetual.csv",
    *QUARTERLY_FILENAMES,
)


def main() -> None:
    """Rebuild processed parquet inputs from restored raw source files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check every source file first: the loop below deletes the bundled runtime
    # parquet inputs, and a missing source would leave the project with no data.
    missing = [
        name for name in REQUIRED_RAW_FILENAMES if not (RAW_DATA_DIR / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot rebuild processed data; missing raw source files under "
            f"{RAW_DATA_DIR}: {', '.join(missing)}"
        )

    for existing_file in PROCESSED_DATA_DIR.iterdir():
        if existing_file.is_file():
            existing_file.unlink()

    write_btc_spot_parquet()
    write_bito_parquet()
    write_perpetual_parquet()
    for filename in QUARTERLY_FILENAMES:
        write_quarterly_parquet(filename)


if __name__ == "__main__":
    main()
