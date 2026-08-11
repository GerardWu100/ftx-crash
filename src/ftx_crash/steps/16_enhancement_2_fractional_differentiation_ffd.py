"""Notebook section: enhancement 2 fractional differentiation ffd."""

# Enhancement 2: Fractional Differentiation
# Using RiskLabAI's find_optimal_ffd_simple() directly

# Prepare the basis series
luna_basis_series = data.loc[
    luna_estimation_start:luna_post_end, "luna_front_basis"
].dropna()
ftx_basis_series = data.loc[
    ftx_estimation_start:ftx_post_end, "ftx_front_basis"
].dropna()

# --- FIX: ROBUST SHIFT FOR LOG-TRANSFORM ---
# Annualized basis can be extremely negative (e.g., -500%) near maturity during crises.
# We calculate a dynamic shift to ensure ALL values are strictly positive (> 1)
# so that np.log() (used inside the FFD function) never fails.

global_min = min(luna_basis_series.min(), ftx_basis_series.min())
shift_val = abs(global_min) + 100  # Dynamic shift + 100% buffer
print(f"Applied Shift Value: {shift_val:.2f}% (Lowest basis was {global_min:.2f}%)")

luna_basis_df = pd.DataFrame({"close": luna_basis_series + shift_val})
ftx_basis_df = pd.DataFrame({"close": ftx_basis_series + shift_val})

print("Fractional Differentiation Analysis (Trend Preserved)")
print("=" * 70)
print(f"LUNA basis series: {len(luna_basis_series)} observations")
print(f"FTX basis series:  {len(ftx_basis_series)} observations")

# Call RiskLabAI's find_optimal_ffd_simple()
print("\nRunning FFD optimization for LUNA...")
luna_ffd_results = diff.find_optimal_ffd_simple(luna_basis_df, p_value_threshold=0.05)

print("Running FFD optimization for FTX...")
ftx_ffd_results = diff.find_optimal_ffd_simple(ftx_basis_df, p_value_threshold=0.05)

print("\nLUNA Basis Series - FFD Results:")
print(luna_ffd_results.to_string())

print("\n" + "-" * 70)
print("\nFTX Basis Series - FFD Results:")
print(ftx_ffd_results.to_string())

# Find optimal d: minimum d where ADF statistic < 95% confidence level
# We add a safety check for empty results
if not luna_ffd_results.empty:
    luna_stationary = luna_ffd_results[
        luna_ffd_results["adfStat"] < luna_ffd_results["95% conf"]
    ]
    luna_optimal_d = luna_stationary.index.min() if len(luna_stationary) > 0 else 1.0
    luna_corr_at_opt = luna_ffd_results.loc[luna_optimal_d, "corr"]
else:
    luna_optimal_d = 1.0
    luna_corr_at_opt = 0.0

if not ftx_ffd_results.empty:
    ftx_stationary = ftx_ffd_results[
        ftx_ffd_results["adfStat"] < ftx_ffd_results["95% conf"]
    ]
    ftx_optimal_d = ftx_stationary.index.min() if len(ftx_stationary) > 0 else 1.0
    ftx_corr_at_opt = ftx_ffd_results.loc[ftx_optimal_d, "corr"]
else:
    print("\nWARNING: FTX FFD Results still empty despite shift. defaulting to d=1.0")
    ftx_optimal_d = 1.0
    ftx_corr_at_opt = 0.0

print("\n" + "=" * 70)
print("OPTIMAL DIFFERENTIATION DEGREE (minimum d for stationarity):")
print(
    f"  LUNA: d* = {luna_optimal_d:.1f} (preserves {luna_corr_at_opt * 100:.1f}% of memory)"
)
print(
    f"  FTX:  d* = {ftx_optimal_d:.1f} (preserves {ftx_corr_at_opt * 100:.1f}% of memory)"
)

# FFD trade-off: memory (correlation) vs stationarity (ADF) at each differentiation degree d.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: LUNA
ax1 = axes[0]
ax1.plot(
    luna_ffd_results.index,
    luna_ffd_results["corr"],
    "o-",
    color="#e74c3c",
    linewidth=2,
    markersize=8,
    label="Memory (Correlation)",
)
ax1.axhline(
    y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% memory threshold"
)
ax1.axvline(
    x=luna_optimal_d,
    color="green",
    linestyle="-",
    linewidth=2,
    alpha=0.7,
    label=f"Optimal d*={luna_optimal_d:.1f}",
)
ax1.set_xlabel("Differentiation Degree (d)", fontsize=11)
ax1.set_ylabel("Correlation with Original", fontsize=11)
ax1.set_title("LUNA Basis: FFD Trade-off", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Mark stationary/non-stationary points
for d_val in luna_ffd_results.index:
    is_stationary = (
        luna_ffd_results.loc[d_val, "adfStat"] < luna_ffd_results.loc[d_val, "95% conf"]
    )
    color = "green" if is_stationary else "red"
    ax1.scatter(
        d_val,
        luna_ffd_results.loc[d_val, "corr"],
        color=color,
        s=100,
        zorder=5,
        edgecolor="black",
    )

# Right: FTX
ax2 = axes[1]
ax2.plot(
    ftx_ffd_results.index,
    ftx_ffd_results["corr"],
    "o-",
    color="#9b59b6",
    linewidth=2,
    markersize=8,
    label="Memory (Correlation)",
)
ax2.axhline(
    y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% memory threshold"
)
ax2.axvline(
    x=ftx_optimal_d,
    color="green",
    linestyle="-",
    linewidth=2,
    alpha=0.7,
    label=f"Optimal d*={ftx_optimal_d:.1f}",
)
ax2.set_xlabel("Differentiation Degree (d)", fontsize=11)
ax2.set_ylabel("Correlation with Original", fontsize=11)
ax2.set_title("FTX Basis: FFD Trade-off", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# Mark stationary/non-stationary points
for d_val in ftx_ffd_results.index:
    is_stationary = (
        ftx_ffd_results.loc[d_val, "adfStat"] < ftx_ffd_results.loc[d_val, "95% conf"]
    )
    color = "green" if is_stationary else "red"
    ax2.scatter(
        d_val,
        ftx_ffd_results.loc[d_val, "corr"],
        color=color,
        s=100,
        zorder=5,
        edgecolor="black",
    )

plt.tight_layout()
save_paper_fig("fig_appendix_ffd_tradeoff")
plt.show()

print("\nGreen = stationary (ADF test passes), Red = non-stationary")
print(
    f"LUNA achieves stationarity at d={luna_optimal_d:.1f}, FTX at d={ftx_optimal_d:.1f}"
)
