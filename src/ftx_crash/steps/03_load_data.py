"""Notebook section: load data."""

import os

# All runtime inputs live under data/processed/ (parquet built by scripts/convert_raw_to_parquet.py).
base_dir = str(PROCESSED_DATA_DIR)

# --- BTC spot (1-minute, UTC) ---
btc_path = os.path.join(base_dir, "BTC_full_1min.parquet")
print(f"Loading BTC Spot Data from: {btc_path}...")

btc_1min = pd.read_parquet(btc_path).set_index('DateTime')
btc_1min.index = btc_1min.index.tz_localize('UTC')

luna_spot_intraday = btc_1min[
    (btc_1min.index >= '2022-04-01') &
    (btc_1min.index <= '2022-06-30')
].copy()

ftx_spot_intraday = btc_1min[
    (btc_1min.index >= '2022-10-01') &
    (btc_1min.index <= '2022-12-31')
].copy()

print(f"Loaded {len(luna_spot_intraday):,} bars for LUNA Crisis.")
print(f"Loaded {len(ftx_spot_intraday):,} bars for FTX Crisis.")

# --- BITO ETF as FTX futures proxy (US/Eastern -> UTC) ---
bito_path = os.path.join(base_dir, "BITO_full_1min_adjsplitdiv.parquet")
print(f"\nLoading Futures Proxy (BITO) from: {bito_path}...")

ftx_futures_proxy = None
if os.path.exists(bito_path):
    bito_1min = pd.read_parquet(bito_path).set_index('DateTime')
    bito_1min.index = bito_1min.index.tz_localize('US/Eastern').tz_convert('UTC')
    ftx_futures_proxy = bito_1min[
        (bito_1min.index >= '2022-10-01') &
        (bito_1min.index <= '2022-12-31')
    ].copy()
    print(f"Loaded {len(ftx_futures_proxy):,} bars for FTX Crisis (Aligned to UTC).")
else:
    print(f"ERROR: BITO file not found at {bito_path}")
