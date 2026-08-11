"""Notebook section: enhancement 1 event based sampling cusum filter."""

# Enhancement 1: CUSUM Filter for Event-Based Sampling
# Sample when significant price movements occur instead of fixed time intervals

# WORKAROUND: RiskLabAI compatibility fix for Timezones
# We perform CUSUM calculations on naive data, then re-localize the results.

# 1. Prepare Data (Strip Timezone)
btc_prices = data["perp_close"].dropna()
btc_prices_naive = btc_prices.copy()
btc_prices_naive.index = btc_prices_naive.index.tz_localize(None)

# 2. Calculate daily volatility for dynamic threshold (on naive data)
# Note: span=20 days roughly
daily_vol_naive = daily_volatility_with_log_returns(btc_prices_naive, span=100)

# Use 1.5x daily volatility as CUSUM threshold (in price units)
# Convert volatility to price threshold: threshold = vol * price
cusum_threshold_naive = (daily_vol_naive * btc_prices_naive).dropna()

# 3. Apply dynamic CUSUM filter (on naive data)
print("Running CUSUM filter (this may take a moment)...")
cusum_events_naive = cusum_filter_events_dynamic_threshold(
    btc_prices_naive, cusum_threshold_naive
)

# 4. Re-localize Events to UTC (Match original data)
cusum_events = cusum_events_naive.tz_localize("UTC")

# Compare event density between time-based and event-based sampling
print("Event-Based Sampling Results (CUSUM Filter)")
print("=" * 60)
print(f"\nTime-based sampling (1-min): {len(data)} observations")
print(f"Event-based sampling (CUSUM): {len(cusum_events)} events")
print(f"Reduction ratio: {len(cusum_events) / len(data) * 100:.2f}% of original data")

# Count events in each crisis period
luna_cusum = cusum_events[
    (cusum_events >= luna_crisis_start) & (cusum_events <= luna_crisis_end)
]
ftx_cusum = cusum_events[
    (cusum_events >= ftx_crisis_start) & (cusum_events <= ftx_crisis_end)
]

# Compare to time-based sampling
luna_hours = (luna_crisis_end - luna_crisis_start).total_seconds() / 3600
ftx_hours = (ftx_crisis_end - ftx_crisis_start).total_seconds() / 3600

print("\nCrisis Period Event Density:")
print(
    f"  LUNA Crisis: {len(luna_cusum)} events in {luna_hours:.0f} hours = {len(luna_cusum) / luna_hours:.2f} events/hour"
)
print(
    f"  FTX Crisis:  {len(ftx_cusum)} events in {ftx_hours:.0f} hours = {len(ftx_cusum) / ftx_hours:.2f} events/hour"
)

# Estimation period event density for comparison
luna_est_cusum = cusum_events[
    (cusum_events >= luna_estimation_start) & (cusum_events <= luna_estimation_end)
]
ftx_est_cusum = cusum_events[
    (cusum_events >= ftx_estimation_start) & (cusum_events <= ftx_estimation_end)
]
luna_est_hours = (luna_estimation_end - luna_estimation_start).total_seconds() / 3600
ftx_est_hours = (ftx_estimation_end - ftx_estimation_start).total_seconds() / 3600

print("\nEstimation Period Event Density:")
print(
    f"  LUNA Est: {len(luna_est_cusum)} events in {luna_est_hours:.0f} hours = {len(luna_est_cusum) / luna_est_hours:.3f} events/hour"
)
print(
    f"  FTX Est:  {len(ftx_est_cusum)} events in {ftx_est_hours:.0f} hours = {len(ftx_est_cusum) / ftx_est_hours:.3f} events/hour"
)

print("\nCrisis-to-Estimation Event Density Ratio:")
try:
    luna_ratio = (len(luna_cusum) / luna_hours) / (len(luna_est_cusum) / luna_est_hours)
    ftx_ratio = (len(ftx_cusum) / ftx_hours) / (len(ftx_est_cusum) / ftx_est_hours)
    print(f"  LUNA: {luna_ratio:.1f}x more events during crisis")
    print(f"  FTX:  {ftx_ratio:.1f}x more events during crisis")
except ZeroDivisionError:
    print("  (Insufficient data in estimation period to calculate ratio)")
