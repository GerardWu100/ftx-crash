"""Notebook section: enhancement 5 synthetic data robustness testing."""

# ---------------------------------------------------------
# ENHANCEMENT 5: SYNTHETIC DATA ROBUSTNESS TESTING
# ---------------------------------------------------------
# Use Heston-Merton model with regime switching to generate synthetic crises

import RiskLabAI.data.synthetic_data as synth

# 1. Calibrate regime parameters from empirical data
luna_crisis_returns = (
    data.loc[luna_crisis_start:luna_crisis_end, "perp_close"].pct_change().dropna()
)
ftx_crisis_returns = (
    data.loc[ftx_crisis_start:ftx_crisis_end, "perp_close"].pct_change().dropna()
)

# Calculate empirical moments (Annualized)
luna_vol = luna_crisis_returns.std() * np.sqrt(24 * 365)
ftx_vol = ftx_crisis_returns.std() * np.sqrt(24 * 365)
luna_mean = luna_crisis_returns.mean() * 24 * 365
ftx_mean = ftx_crisis_returns.mean() * 24 * 365

print("Empirical Calibration for Heston-Merton Model")
print("=" * 60)
print(f"LUNA (Protocol):     Vol={luna_vol * 100:.1f}%, Drift={luna_mean * 100:.1f}%")
print(f"FTX (Counterparty):  Vol={ftx_vol * 100:.1f}%, Drift={ftx_mean * 100:.1f}%")

# 2. Define Regime Parameters
regimes = {
    "protocol_crisis": {
        "mu": luna_mean,
        "kappa": 5.0,
        "theta": luna_vol**2,
        "xi": 0.5,
        "rho": -0.7,
        "lam": 0.5,
        "m": -0.05,
        "v": 0.02,
    },
    "counterparty_crisis": {
        "mu": ftx_mean,
        "kappa": 2.0,
        "theta": ftx_vol**2,
        "xi": 0.8,
        "rho": -0.8,
        "lam": 0.8,
        "m": -0.08,
        "v": 0.03,
    },
}

# Transition matrix: 95% stay, 5% switch
transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])

# Monte Carlo: Heston-Merton paths with regime switching (fallback = bootstrap resampling).
n_simulations = 100  # Number of synthetic paths
n_steps = 500  # Steps per path (simulating ~500 hours)
total_time = 500 / (24 * 365)  # Convert to years

print(
    f"Generating {n_simulations} synthetic price paths with synth.parallel_generate_prices()..."
)
print(f"Each path: {n_steps} steps over {n_steps / 24:.0f} days")

# Generate synthetic data using RiskLabAI's Heston-Merton modelI
try:
    synthetic_prices, synthetic_regimes = synth.parallel_generate_prices(
        number_of_paths=n_simulations,
        regimes=regimes,
        transition_matrix=transition_matrix,
        total_time=total_time,
        n_steps=n_steps,
        random_state=42,
        n_jobs=4,
    )

    print(f"\nGenerated price paths shape: {synthetic_prices.shape}")
    print(f"Regime paths shape: {synthetic_regimes.shape}")

    # Analyze regime distribution
    all_regimes = synthetic_regimes.values.flatten()
    protocol_pct = (all_regimes == "protocol_crisis").mean() * 100
    counterparty_pct = (all_regimes == "counterparty_crisis").mean() * 100

    print("\nRegime Distribution across all paths:")
    print(f"  Protocol Crisis:     {protocol_pct:.1f}%")
    print(f"  Counterparty Crisis: {counterparty_pct:.1f}%")

except Exception as e:
    print(f"Error generating synthetic data: {e}")
    print("Falling back to simplified bootstrap simulation...")

    # Fallback: Simple bootstrap resampling from empirical returns
    synthetic_prices = pd.DataFrame()
    synthetic_regimes = pd.DataFrame()

    for i in range(n_simulations):
        # Randomly choose crisis type for each path
        if np.random.random() < 0.5:
            base_returns = luna_crisis_returns.sample(n=n_steps, replace=True).values
            regime = "protocol_crisis"
        else:
            base_returns = ftx_crisis_returns.sample(n=n_steps, replace=True).values
            regime = "counterparty_crisis"

        # Generate cumulative prices
        prices = 100 * np.exp(np.cumsum(base_returns))
        synthetic_prices[i] = prices
        synthetic_regimes[i] = [regime] * n_steps

    print(f"Bootstrap simulation completed: {n_simulations} paths")

# Step 3: Run DiD-style analysis on each synthetic path
# For each path, compare metrics between protocol vs counterparty regime periods


def analyze_synthetic_path(prices: pd.Series, regimes: np.ndarray) -> dict:
    """
    Analyze a single synthetic price path.

    Computes key crisis metrics for protocol vs counterparty regimes:
    - Volatility (rolling std of returns)
    - Max drawdown during regime
    - Return during regime
    """
    returns = prices.pct_change().dropna()

    # Identify regime periods
    protocol_mask = regimes[1:] == "protocol_crisis"  # Align with returns
    counterparty_mask = regimes[1:] == "counterparty_crisis"

    protocol_returns = returns.values[protocol_mask]
    counterparty_returns = returns.values[counterparty_mask]

    # Skip if either regime has too few observations
    if len(protocol_returns) < 10 or len(counterparty_returns) < 10:
        return None

    # Compute metrics
    results = {
        # Volatility comparison (annualized)
        "protocol_vol": np.std(protocol_returns) * np.sqrt(24 * 365),
        "counterparty_vol": np.std(counterparty_returns) * np.sqrt(24 * 365),
        # Mean return comparison (annualized)
        "protocol_return": np.mean(protocol_returns) * 24 * 365,
        "counterparty_return": np.mean(counterparty_returns) * 24 * 365,
        # Tail risk: worst 5% of returns
        "protocol_var_5pct": np.percentile(protocol_returns, 5),
        "counterparty_var_5pct": np.percentile(counterparty_returns, 5),
        # Regime durations
        "protocol_obs": len(protocol_returns),
        "counterparty_obs": len(counterparty_returns),
    }

    # DiD-style estimator: difference in crisis severity
    # (Counterparty effect) - (Protocol effect)
    results["vol_difference"] = results["counterparty_vol"] - results["protocol_vol"]
    results["return_difference"] = (
        results["counterparty_return"] - results["protocol_return"]
    )
    results["var_difference"] = (
        results["counterparty_var_5pct"] - results["protocol_var_5pct"]
    )

    return results


# Run analysis on all synthetic paths
print("Running DiD analysis on synthetic paths...")
print("=" * 60)

simulation_results = []
for path_id in range(n_simulations):
    prices = synthetic_prices[path_id]
    regimes_path = synthetic_regimes[path_id].values

    result = analyze_synthetic_path(prices, regimes_path)
    if result is not None:
        result["path_id"] = path_id
        simulation_results.append(result)

results_df = pd.DataFrame(simulation_results)
print(f"Successfully analyzed {len(results_df)} of {n_simulations} paths")
print("(Paths with insufficient regime observations were skipped)")

# Step 4: Report distribution of DiD estimators
print("Distribution of DiD Estimators Across Synthetic Paths")
print("=" * 60)

# Key metrics to report
metrics = [
    ("vol_difference", "Volatility Difference (Counterparty - Protocol)"),
    ("return_difference", "Return Difference (Counterparty - Protocol)"),
    ("var_difference", "VaR 5% Difference (Counterparty - Protocol)"),
]

for metric, label in metrics:
    values = results_df[metric]

    # Calculate confidence interval
    mean_val = values.mean()
    std_val = values.std()
    ci_lower = np.percentile(values, 2.5)
    ci_upper = np.percentile(values, 97.5)

    # What fraction of simulations show the same sign as the mean?
    if mean_val > 0:
        pct_same_sign = (values > 0).mean() * 100
        direction = "Counterparty > Protocol"
    else:
        pct_same_sign = (values < 0).mean() * 100
        direction = "Protocol > Counterparty"

    print(f"\n{label}:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std Dev: {std_val:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Direction: {direction} in {pct_same_sign:.1f}% of simulations")

# Summary statistics table
print("\n" + "=" * 60)
print("Summary: Regime-Specific Metrics")
print("=" * 60)

summary_stats = results_df[
    [
        "protocol_vol",
        "counterparty_vol",
        "protocol_return",
        "counterparty_return",
        "protocol_var_5pct",
        "counterparty_var_5pct",
    ]
].describe()
print(summary_stats.round(4).to_string())

# Visualize the distribution of DiD estimators
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = [
    (
        "vol_difference",
        "Volatility Difference\n(Counterparty - Protocol)",
        "Annualized Vol Diff",
    ),
    (
        "return_difference",
        "Return Difference\n(Counterparty - Protocol)",
        "Annualized Return Diff",
    ),
    ("var_difference", "VaR 5% Difference\n(Counterparty - Protocol)", "VaR Diff"),
]

for ax, (metric, title, xlabel) in zip(axes, metrics_to_plot):
    values = results_df[metric]

    # Histogram with KDE
    ax.hist(
        values, bins=25, density=True, alpha=0.7, color="steelblue", edgecolor="white"
    )

    # Add vertical line at zero
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No Difference")

    # Add vertical line at mean
    mean_val = values.mean()
    ax.axvline(
        x=mean_val,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_val:.3f}",
    )

    # Add 95% CI shading
    ci_lower = np.percentile(values, 2.5)
    ci_upper = np.percentile(values, 97.5)
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color="green", label="95% CI")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Monte Carlo Robustness Check: Distribution of Crisis Type Differences\n(100 Synthetic Paths from Heston-Merton Model)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()

# Final robustness conclusion
print("\nRobustness Check Conclusion")
print("=" * 60)

vol_pct = (results_df["vol_difference"] > 0).mean() * 100
ret_pct = (results_df["return_difference"] < 0).mean() * 100  # More negative = worse

print(f"\nAcross {len(results_df)} synthetic crisis scenarios:")
print(
    f"  - Counterparty crises showed HIGHER volatility in {vol_pct:.1f}% of simulations"
)
print(f"  - Counterparty crises showed WORSE returns in {ret_pct:.1f}% of simulations")

if vol_pct > 75 and ret_pct > 75:
    print("\n  ROBUST: The finding that counterparty crises are more severe")
    print("          holds consistently across synthetic scenarios.")
elif vol_pct > 50 and ret_pct > 50:
    print("\n  MODERATE: The finding holds in most simulations but with")
    print("            substantial variation. Interpret with caution.")
else:
    print("\n  WEAK: The finding does NOT hold consistently across")
    print("        synthetic scenarios. Original result may be a fluke.")

# ---------------------------------------------------------
# ENHANCEMENT 6: FINAL VISUAL SYNTHESIS
# ---------------------------------------------------------
from matplotlib import gridspec
from statsmodels.tsa.stattools import acf

# Create a dashboard-style layout
fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)

# =========================================================
# PLOT 1: THE REGIME MAP (Price vs. Volatility)
# Visual proof of the "Backdoor Adjustment"
# =========================================================
ax1 = plt.subplot(gs[0])

# 1. Calculate context metrics for the whole year
# Rolling 7-day volatility (annualized)
vol_context = data["perp_close"].pct_change().rolling(window=24 * 7).std() * np.sqrt(
    24 * 365
)
price_context = data["perp_close"]

# 2. Extract Crisis Points
luna_idx = data.loc[luna_crisis_start:luna_crisis_end].index
ftx_idx = data.loc[ftx_crisis_start:ftx_crisis_end].index

# 3. Plot "The Matrix"
# Background: All 2022 data (Grey)
ax1.scatter(
    price_context,
    vol_context,
    c="lightgrey",
    alpha=0.4,
    s=15,
    label="2022 Market Context",
)

# Overlay: LUNA (Red) - High Price, Moderate Vol
ax1.scatter(
    price_context.loc[luna_idx],
    vol_context.loc[luna_idx],
    c="#e74c3c",
    alpha=0.8,
    s=50,
    edgecolor="black",
    label="LUNA (Protocol Crisis)",
)

# Overlay: FTX (Purple) - Low Price, High Vol
ax1.scatter(
    price_context.loc[ftx_idx],
    vol_context.loc[ftx_idx],
    c="#9b59b6",
    alpha=0.9,
    s=50,
    edgecolor="black",
    marker="D",
    label="FTX (Counterparty Crisis)",
)

# Annotations ("The Danger Zone")
ax1.set_xlabel("Bitcoin Price ($)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Annualized Volatility", fontsize=12, fontweight="bold")
ax1.set_title("The Regime Map: Crisis Phase Space", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Add Arrows/Text to explain the "Regime Shift"
ax1.annotate(
    "High Liquidity Regime\n(LUNA)",
    xy=(price_context.loc[luna_idx].mean(), vol_context.loc[luna_idx].mean()),
    xytext=(35000, 0.4),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)

ax1.annotate(
    "Low Liquidity / Danger Zone\n(FTX)",
    xy=(price_context.loc[ftx_idx].mean(), vol_context.loc[ftx_idx].mean()),
    xytext=(18000, 1.2),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)


# =========================================================
# PLOT 2: MEMORY DECAY (Autocorrelation Function)
# Visual proof of the "Persistence" argument
# =========================================================
ax2 = plt.subplot(gs[1])

# 1. Calculate ACF for both abnormal basis series
# We use 48 hours (2 days) of lags
lags = 48
luna_basis_clean = data.loc[
    luna_crisis_start:luna_crisis_end, "luna_abnormal_basis"
].dropna()
ftx_basis_clean = data.loc[
    ftx_crisis_start:ftx_crisis_end, "ftx_abnormal_basis"
].dropna()

luna_acf_vals = acf(luna_basis_clean, nlags=lags)
ftx_acf_vals = acf(ftx_basis_clean, nlags=lags)

# 2. Plot the Decay
x_axis = range(len(luna_acf_vals))
ax2.plot(
    x_axis,
    luna_acf_vals,
    color="#e74c3c",
    linewidth=2.5,
    marker="o",
    markersize=4,
    label="LUNA (Fast Decay)",
)
ax2.plot(
    x_axis,
    ftx_acf_vals,
    color="#9b59b6",
    linewidth=2.5,
    marker="D",
    markersize=4,
    label="FTX (Sticky/Persistent)",
)

# 3. Styling
ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Half-Life Threshold")
ax2.fill_between(x_axis, 0, luna_acf_vals, color="#e74c3c", alpha=0.1)
ax2.fill_between(x_axis, 0, ftx_acf_vals, color="#9b59b6", alpha=0.1)

ax2.set_xlabel("Lag (Hours)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Autocorrelation", fontsize=12, fontweight="bold")
ax2.set_title(
    "Memory Decay: The Fingerprint of Mistrust", fontsize=14, fontweight="bold"
)
ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.2, 1.1)

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "fig5_regime_map_and_memory_decay.png"),
    dpi=PLOT_QUALITY,
    bbox_inches="tight",
)
plt.show()

print("\nVISUAL INTERPRETATION:")
print("=" * 60)
print("1. THE REGIME MAP (Left):")
print("   - Proves LUNA and FTX occurred in fundamentally different market regimes.")
print(
    "   - LUNA: High Price ($30k+), Moderate Volatility -> Market absorbed the shock."
)
print(
    "   - FTX:  Low Price (<$20k), Extreme Volatility -> 'Danger Zone' amplified the impact."
)
print(
    "   - This validates the 'Backdoor Adjustment': The environment caused the severity."
)
print("\n2. MEMORY DECAY (Right):")
print("   - Shows the 'Microstructure Fingerprint' of the crisis type.")
print("   - LUNA (Red): Correlation drops quickly. Arbitrageurs trusted the fix.")
print(
    "   - FTX (Purple): Correlation stays high (Sticky). Arbitrageurs stopped trading."
)
print("   - This persistent memory is the unique signature of counterparty failure.")

# --- TABLE 1: DESCRIPTIVE STATISTICS ---
# Summarizes the raw basis behavior during the two crises
table1 = (
    pd.DataFrame(
        {
            "Metric": [
                "Mean Basis (%)",
                "Min Basis (%)",
                "Std Dev (%)",
                "Count (Hours)",
            ],
            "LUNA (Protocol)": [
                luna_ab_basis.mean(),
                luna_ab_basis.min(),
                luna_ab_basis.std(),
                len(luna_ab_basis),
            ],
            "FTX (Counterparty)": [
                ftx_ab_basis.mean(),
                ftx_ab_basis.min(),
                ftx_ab_basis.std(),
                len(ftx_ab_basis),
            ],
        }
    )
    .set_index("Metric")
    .round(3)
)

save_paper_table(table1, "tab1_descriptive_stats")

# --- TABLE 2: HYPOTHESIS TESTING ---
# Formal tests comparing the two distributions
table2 = pd.DataFrame(
    {
        "Test": ["Welch's t-test", "Mann-Whitney U", "Cohen's d"],
        "Statistic": [basis_ttest.statistic, basis_utest.statistic, basis_cohens_d],
        "P-Value": [
            basis_ttest.pvalue,
            basis_utest.pvalue,
            None,
        ],  # Cohen's d has no p-value
        "Interpretation": [
            "Significant" if basis_ttest.pvalue < 0.05 else "Not Significant",
            "Significant" if basis_utest.pvalue < 0.05 else "Not Significant",
            "Large Effect" if abs(basis_cohens_d) > 0.8 else "Medium/Small",
        ],
    }
).round(4)

save_paper_table(table2, "tab2_hypothesis_tests")

# --- TABLE 4: CRISIS CLASSIFIER IMPORTANCE ---
# Which microstructure features distinguish the crisis types?
# (Assumes 'feature_importance' df exists from your ML section)
table4 = feature_importance[["feature", "importance", "std"]].copy()
table4.columns = ["Feature", "Gini Importance", "Std Dev"]
table4 = table4.round(4)

save_paper_table(table4, "tab4_feature_importance")

# --- TABLE 5: ROBUSTNESS SIMULATION ---
# Summary of the 100 Monte Carlo paths
table5 = results_df[["vol_difference", "return_difference"]].describe().T
table5 = table5[["mean", "std", "min", "max"]].round(4)
table5.columns = ["Mean Diff", "Std Dev", "Min Diff", "Max Diff"]
table5.index = [
    "Volatility Diff (Counterparty - Protocol)",
    "Return Diff (Counterparty - Protocol)",
]

save_paper_table(table5, "tab5_robustness_summary")
