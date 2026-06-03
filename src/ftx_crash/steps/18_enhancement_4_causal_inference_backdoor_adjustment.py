"""Notebook section: enhancement 4 causal inference backdoor adjustment."""

# --- FIX: CAUSAL INFERENCE ON STATIONARY DATA ---
# We must run the regression on the FFD-transformed basis to satisfy stationarity assumptions.

from RiskLabAI.data.differentiation.differentiation import fractional_difference_fixed

# 1. Generate Stationary Series using the optimal d found earlier
# Use the SAME shift_val calculated in the previous step (e.g., ~150-2000)
# to ensure the data range matches what you calibrated on.
luna_ffd_series = fractional_difference_fixed(
    pd.DataFrame({'close': data['luna_abnormal_basis'].dropna() + shift_val}), 
    degree=luna_optimal_d  # Correct parameter name is 'degree'
)

ftx_ffd_series = fractional_difference_fixed(
    pd.DataFrame({'close': data['ftx_abnormal_basis'].dropna() + shift_val}), 
    degree=ftx_optimal_d   # Correct parameter name is 'degree'
)

# 2. Align Data for Regression
def prepare_robust_causal_dataset(original_data, ffd_series, crisis_start, crisis_end, crisis_label):
    # Slice to crisis window
    crisis_slice = original_data.loc[crisis_start:crisis_end].copy()
    
    # Merge FFD data (ensure index alignment)
    # The FFD series index might be slightly shorter due to windowing, so we reindex
    crisis_slice['ffd_basis'] = ffd_series['close'].reindex(crisis_slice.index)
    
    # Controls
    returns = crisis_slice['perp_close'].pct_change()
    volatility = returns.rolling(24).std() * np.sqrt(24) # 24h vol
    
    df = pd.DataFrame({
        'ffd_basis': crisis_slice['ffd_basis'], # The stationary target
        'btc_volatility': volatility,
        'log_price': np.log(crisis_slice['perp_close']),
        'crisis_type': crisis_label
    }).dropna()
    return df

luna_robust = prepare_robust_causal_dataset(data, luna_ffd_series, luna_crisis_start, luna_crisis_end, 0)
ftx_robust = prepare_robust_causal_dataset(data, ftx_ffd_series, ftx_crisis_start, ftx_crisis_end, 1)
combined_robust = pd.concat([luna_robust, ftx_robust], ignore_index=True)

# 3. Run Robust Regression
X_robust = sm.add_constant(combined_robust[['crisis_type', 'btc_volatility', 'log_price']])
y_robust = combined_robust['ffd_basis']
model_robust = sm.OLS(y_robust, X_robust).fit()

print("\n=== ROBUST CAUSAL INFERENCE (ON STATIONARY FFD DATA) ===")
print(model_robust.summary())

# --- Comparison Logic (Using Robust Model) ---
print("\n" + "=" * 70)
print("ROBUST MODEL INTERPRETATION")
print("=" * 70)
print(f"FTX Effect (Adjusted): {model_robust.params['crisis_type']:.4f}")
print(f"P-value:               {model_robust.pvalues['crisis_type']:.2e}")

if model_robust.pvalues['crisis_type'] < 0.05:
    print("\nCONCLUSION: Even after enforcing stationarity (FFD) and controlling for regimes,")
    print("the crisis type had a statistically significant impact on the basis.")
else:
    print("\nCONCLUSION: After enforcing stationarity (FFD), the difference between crisis types")
    print("is no longer statistically significant. The 'dislocation' was likely driven")
    print("by the market regime (volatility/price) rather than the nature of the crisis itself.")

# --- FIX: SAVE ROBUST MODEL RESULTS ---
# We use 'model_robust' because we replaced 'model_controls' with the FFD version.

causal_table = pd.DataFrame({
    'Coefficient': model_robust.params,
    'Std Error': model_robust.bse,
    't-statistic': model_robust.tvalues,
    'P-value': model_robust.pvalues
})

print("\n--- Saving Causal Inference Results ---")
# Saving as the main table for the paper
save_table(causal_table, 'tab3_causal_regression_results')

# Interpretation driven by the fitted robust model (not hard-coded notebook coefficients).
print("Interpretation of Backdoor Adjustment Results")
print("=" * 70)
crisis_coef = model_robust.params['crisis_type']
crisis_p = model_robust.pvalues['crisis_type']
print(f"  crisis_type coefficient (FTX vs LUNA, FFD-adjusted): {crisis_coef:.4f} (p={crisis_p:.2e})")
print(f"  log_price:      {model_robust.params['log_price']:.4f}")
print(f"  btc_volatility: {model_robust.params['btc_volatility']:.4f}")
print(
    "\nAfter fractional differentiation and controls for price/volatility regimes, "
    "read the crisis_type sign as incremental FTX vs LUNA basis dislocation "
    "at comparable market conditions."
)

# Confounder distributions used in the backdoor adjustment.
# We use 'combined_robust' which was created in the previous step.
# We plot the 2 key controls used in the regression: Price and Volatility.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjusted to 2 columns

# Confounder distributions by crisis
variables = ['log_price', 'btc_volatility']
titles = ['Log Price Distribution (Regime)', 'Volatility Distribution (Regime)']

for i, (var, title) in enumerate(zip(variables, titles)):
    luna_vals = combined_robust[combined_robust['crisis_type']==0][var]
    ftx_vals = combined_robust[combined_robust['crisis_type']==1][var]

    axes[i].hist(luna_vals, alpha=0.6, label='LUNA', color='#e74c3c', bins=20, density=True)
    axes[i].hist(ftx_vals, alpha=0.6, label='FTX', color='#9b59b6', bins=20, density=True)
    axes[i].set_xlabel(var, fontsize=11)
    axes[i].set_ylabel('Density', fontsize=11)
    axes[i].legend()
    axes[i].set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('Confounder Distributions: LUNA vs FTX Crisis Periods\n(Visual Proof of Regime Difference)', 
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
save_paper_fig('fig_appendix_confounder_distributions')
plt.show()
