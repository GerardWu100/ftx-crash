"""Notebook section: market overview."""

# Hourly BTC perpetual prices for the 2022 overview chart (separate from 1-min basis panel).
data_dir = PROCESSED_DATA_DIR
btc_perp = pd.read_parquet(data_dir / 'DA-16_BTC_USDT_OPEN_HIGH_LOW_LAST_PRICE_hourly_2019-2025_binance_perpetual.parquet')

btc_perp['datetime'] = pd.to_datetime(btc_perp['datetime'])
btc_perp = btc_perp.set_index('datetime').sort_index()

# Focus on 2022 data covering both crises
btc_2022 = btc_perp.loc['2022-01-01':'2023-01-01', 'close'].copy()

# Overview chart windows (slightly wider than event-study crisis windows in step 04).
luna_crisis_start = '2022-05-07'
luna_crisis_end = '2022-05-15'
ftx_crisis_start = '2022-11-06'
ftx_crisis_end = '2022-11-14'

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(btc_2022.index, btc_2022, linewidth=1.5, color='#2E86AB', label='BTC Price')

ax.axvspan(pd.to_datetime(luna_crisis_start), pd.to_datetime(luna_crisis_end),
           alpha=0.25, color='red', label='LUNA Collapse (May 2022)')
ax.axvspan(pd.to_datetime(ftx_crisis_start), pd.to_datetime(ftx_crisis_end),
           alpha=0.25, color='orange', label='FTX Collapse (Nov 2022)')

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('BTC Price (USDT)', fontsize=12, fontweight='bold')
ax.set_title('Bitcoin Price During 2022 Crypto Market Crises\nTerra/LUNA Collapse vs. FTX Exchange Failure',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_btc_price_trajectory.png'), dpi=PLOT_QUALITY, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Market Crash Summary Statistics:")
print("=" * 60)

print(f"\nLUNA Crisis (May 7-15, 2022):")
luna_data = btc_2022.loc[luna_crisis_start:luna_crisis_end]
print(f"  Start price: ${luna_data.iloc[0]:,.2f}")
print(f"  Low price: ${luna_data.min():,.2f}")
print(f"  Price decline: {((luna_data.min() - luna_data.iloc[0]) / luna_data.iloc[0] * 100):.2f}%")

print(f"\nFTX Crisis (Nov 6-14, 2022):")
ftx_data = btc_2022.loc[ftx_crisis_start:ftx_crisis_end]
print(f"  Start price: ${ftx_data.iloc[0]:,.2f}")
print(f"  Low price: ${ftx_data.min():,.2f}")
print(f"  Price decline: {((ftx_data.min() - ftx_data.iloc[0]) / ftx_data.iloc[0] * 100):.2f}%")

print(f"\nOverall 2022 BTC Performance:")
print(f"  Year start (Jan 1): ${btc_2022.iloc[0]:,.2f}")
print(f"  Year end (Dec 31): ${btc_2022.iloc[-1]:,.2f}")
print(f"  Annual return: {((btc_2022.iloc[-1] - btc_2022.iloc[0]) / btc_2022.iloc[0] * 100):.2f}%")
print(f"  Max price: ${btc_2022.max():,.2f}")
print(f"  Min price: ${btc_2022.min():,.2f}")
