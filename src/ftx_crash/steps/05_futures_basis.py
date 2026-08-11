"""Notebook section: futures basis."""


def calculate_basis(futures_price, spot_price, days_to_maturity):
    """Annualized futures basis in percent: ((F - S) / S) * (365 / T) * 100.

    Parameters
    ----------
    futures_price, spot_price
        Price series (aligned index).
    days_to_maturity
        Calendar days until contract expiry; floored at 1 to avoid division by zero.
    """
    days_to_expiry = np.maximum(days_to_maturity, 1)
    return ((futures_price - spot_price) / spot_price) * (365 / days_to_expiry) * 100


# Quarterly contract expiry dates (last Friday of quarter), UTC for alignment with spot data.
maturity_dates = {
    "2022-06": pd.Timestamp("2022-06-24", tz="UTC"),
    "2022-09": pd.Timestamp("2022-09-30", tz="UTC"),
    "2022-12": pd.Timestamp("2022-12-30", tz="UTC"),
    "2023-03": pd.Timestamp("2023-03-31", tz="UTC"),
}

print("Maturity dates:")
for contract, date in maturity_dates.items():
    print(f"  {contract}: {date.date()}")

print("--- LUNA Crisis Data (Spot) ---")
print(f"Shape: {luna_spot_intraday.shape}")
print(luna_spot_intraday[["Close", "Volume"]].head())
print(
    f"Date Range: {luna_spot_intraday.index.min()} to {luna_spot_intraday.index.max()}"
)

print("\n--- FTX Crisis Data (Spot) ---")
print(f"Shape: {ftx_spot_intraday.shape}")
print(ftx_spot_intraday[["Close", "Volume"]].head())
print(f"Date Range: {ftx_spot_intraday.index.min()} to {ftx_spot_intraday.index.max()}")

print("\n--- FTX Crisis Data (Futures Proxy - BITO) ---")
if ftx_futures_proxy is not None:
    print(f"Shape: {ftx_futures_proxy.shape}")
    print(ftx_futures_proxy[["Close", "Volume"]].head())
    print(
        f"Date Range: {ftx_futures_proxy.index.min()} to {ftx_futures_proxy.index.max()}"
    )
else:
    print("FTX Futures Proxy not loaded.")
