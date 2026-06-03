"""Notebook section: front month basis and term structure."""

# Rebuild the combined LUNA/FTX panel used by event-study steps (uses calculate_basis from step 05).
old_data_dir = PROCESSED_DATA_DIR

# Hourly Binance quarterly futures for LUNA term-structure legs.
f_jun = pd.read_parquet(old_data_dir / 'DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2022-03_2022-06_binance_quarterly.parquet')
f_sep = pd.read_parquet(old_data_dir / 'DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2022-06_2022-09_binance_quarterly.parquet')

for futures_df in [f_jun, f_sep]:
    futures_df['datetime'] = pd.to_datetime(futures_df['datetime']).dt.tz_localize('UTC')
    futures_df.set_index('datetime', inplace=True)
    futures_df.sort_index(inplace=True)

# LUNA: 1-min spot + forward-filled hourly futures.
luna_df = pd.DataFrame(index=luna_spot_intraday.index)
luna_df['perp_close'] = luna_spot_intraday['Close']
luna_df['fut_jun'] = f_jun['close'].reindex(luna_df.index, method='ffill')
luna_df['fut_sep'] = f_sep['close'].reindex(luna_df.index, method='ffill')

days_jun = (maturity_dates['2022-06'] - luna_df.index).days
days_sep = (maturity_dates['2022-09'] - luna_df.index).days

luna_df['basis_2022-06'] = calculate_basis(luna_df['fut_jun'], luna_df['perp_close'], days_jun)
luna_df['basis_2022-09'] = calculate_basis(luna_df['fut_sep'], luna_df['perp_close'], days_sep)
luna_df['luna_slope'] = luna_df['basis_2022-09'] - luna_df['basis_2022-06']
luna_df['luna_front_basis'] = luna_df['basis_2022-06']

# FTX: 1-min spot + BITO proxy (no second futures leg -> slope stays NaN).
ftx_df = pd.DataFrame(index=ftx_spot_intraday.index)
ftx_df['perp_close'] = ftx_spot_intraday['Close']
ftx_df['bito_close'] = ftx_futures_proxy['Close'].reindex(ftx_df.index, method='ffill')

days_dec = (maturity_dates['2022-12'] - ftx_df.index).days
ftx_df['ftx_front_basis'] = calculate_basis(ftx_df['bito_close'], ftx_df['perp_close'], days_dec)
ftx_df['ftx_slope'] = np.nan

data = pd.concat([luna_df, ftx_df])

print("Hybrid Data Object Reconstructed.")
print(f"Total Rows: {len(data)}")
print(f"LUNA Basis Points: {data['luna_front_basis'].count()}")
print(f"FTX Basis Points:  {data['ftx_front_basis'].count()}")
print("-" * 40)

luna_sample = data['luna_front_basis'].dropna()
ftx_sample = data['ftx_front_basis'].dropna()

desc_stats = pd.DataFrame({
    'Metric': ['Mean (%)', 'Std Dev (%)', 'Min (%)', 'Max (%)', 'Count'],
    'LUNA (Protocol)': [
        luna_sample.mean(), luna_sample.std(), luna_sample.min(), luna_sample.max(), len(luna_sample)
    ],
    'FTX (Counterparty)': [
        ftx_sample.mean(), ftx_sample.std(), ftx_sample.min(), ftx_sample.max(), len(ftx_sample)
    ]
}).set_index('Metric')

print("\n--- Saving Descriptive Statistics ---")
save_table(desc_stats, 'tab1_descriptive_statistics')

print(f"\nFront-Month Basis During Crisis:")
print(f"\n  Terra/LUNA (May 2022) [Source: BTC Spot vs Binance Futures]:")
print(f"    Mean: {luna_sample.mean():7.2f}%  |  Std: {luna_sample.std():5.2f}%")
print(f"    Min:  {luna_sample.min():7.2f}%  |  Max: {luna_sample.max():6.2f}%")

print(f"\n  FTX (Nov 2022) [Source: BTC Spot vs BITO ETF]:")
print(f"    Mean: {ftx_sample.mean():7.2f}%  |  Std: {ftx_sample.std():5.2f}%")
print(f"    Min:  {ftx_sample.min():7.2f}%  |  Max:  {ftx_sample.max():6.2f}%")

fig, ax = plt.subplots(figsize=(14, 6))

luna_period = data.loc['2022-03':'2022-07', 'luna_front_basis']
ax.plot(luna_period.index, luna_period, color='#e74c3c', linewidth=1.5, label='Basis during Terra/LUNA')

ftx_period = data.loc['2022-09':'2022-12', 'ftx_front_basis']
ax.plot(ftx_period.index, ftx_period, color='#9b59b6', linewidth=1.5, label='Basis during FTX')

ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.set_ylabel('Annualized Basis (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_title('Historical Basis Dislocation: The "Negative Spikes" of 2022', fontsize=14, fontweight='bold')
ax.legend(frameon=True, framealpha=0.9, loc='lower left')
ax.grid(True, alpha=0.3)
ax.axvspan(luna_crisis_start, luna_crisis_end, color='red', alpha=0.1)
ax.axvspan(ftx_crisis_start, ftx_crisis_end, color='purple', alpha=0.1)

save_paper_fig('fig1b_raw_basis_history')
plt.show()
