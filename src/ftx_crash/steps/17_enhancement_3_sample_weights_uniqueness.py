"""Notebook section: enhancement 3 sample weights uniqueness."""

# Enhancement 3: Sample Weights for Non-IID Correction
# Using RiskLabAI's expand_label_for_meta_labeling() and calculate_average_uniqueness() directly

# Step 1: Prepare event timestamps (index=start, value=end)
# Each basis observation spans from observation time to contract maturity
luna_basis_data = data.loc[
    luna_estimation_start:luna_crisis_end, "luna_front_basis"
].dropna()
ftx_basis_data = data.loc[
    ftx_estimation_start:ftx_crisis_end, "ftx_front_basis"
].dropna()

luna_timestamps = pd.Series(index=luna_basis_data.index, data=maturity_dates["2022-06"])
luna_timestamps = luna_timestamps[luna_timestamps.index < maturity_dates["2022-06"]]

ftx_timestamps = pd.Series(index=ftx_basis_data.index, data=maturity_dates["2022-12"])
ftx_timestamps = ftx_timestamps[ftx_timestamps.index < maturity_dates["2022-12"]]

max_uniqueness_events = int(OVERRIDES.get("max_uniqueness_events", 1000))


def reduce_event_count(timestamps, max_events):
    """Evenly sample event timestamps so the indicator matrix fits in memory."""
    if len(timestamps) <= max_events:
        return timestamps
    sample_positions = np.linspace(0, len(timestamps) - 1, num=max_events, dtype=int)
    reduced = timestamps.iloc[sample_positions]
    print(
        f"Reducing event count from {len(timestamps)} to {len(reduced)} for uniqueness calculation."
    )
    return reduced


luna_timestamps = reduce_event_count(luna_timestamps, max_uniqueness_events)
ftx_timestamps = reduce_event_count(ftx_timestamps, max_uniqueness_events)

print("Sample Uniqueness Weights (using RiskLabAI functions)")
print("=" * 60)
print(
    f"LUNA: {len(luna_timestamps)} events -> maturity {maturity_dates['2022-06'].date()}"
)
print(
    f"FTX:  {len(ftx_timestamps)} events -> maturity {maturity_dates['2022-12'].date()}"
)

# Step 2: Calculate concurrency using expand_label_for_meta_labeling()
# This counts active events at each timestamp
print("\nCalculating concurrency with expand_label_for_meta_labeling()...")
luna_concurrency = expand_label_for_meta_labeling(
    close_index=luna_basis_data.index,
    timestamp=luna_timestamps,
    molecule=luna_timestamps.index,
)
ftx_concurrency = expand_label_for_meta_labeling(
    close_index=ftx_basis_data.index,
    timestamp=ftx_timestamps,
    molecule=ftx_timestamps.index,
)

print(
    f"LUNA concurrency: min={luna_concurrency.min():.0f}, max={luna_concurrency.max():.0f}"
)
print(
    f"FTX concurrency:  min={ftx_concurrency.min():.0f}, max={ftx_concurrency.max():.0f}"
)


# Step 3: Create indicator matrix for calculate_average_uniqueness()
# Indicator matrix: rows=timestamps, columns=events, value=1 if event active
def create_indicator_matrix(close_index, timestamps):
    """Create indicator matrix (T x N) for uniqueness calculation."""
    ind_matrix = pd.DataFrame(
        0,
        index=close_index,
        columns=range(len(timestamps)),
        dtype=np.uint8,
    )
    for i, (t0, t1) in enumerate(timestamps.items()):
        if pd.notna(t1) and t0 in ind_matrix.index:
            valid_idx = ind_matrix.index[
                (ind_matrix.index >= t0) & (ind_matrix.index <= t1)
            ]
            ind_matrix.loc[valid_idx, i] = 1
    return ind_matrix


luna_ind_matrix = create_indicator_matrix(luna_basis_data.index, luna_timestamps)
ftx_ind_matrix = create_indicator_matrix(ftx_basis_data.index, ftx_timestamps)

# Step 4: Calculate average uniqueness using calculate_average_uniqueness()
# This computes: u_i = (1/|I_i|) * sum(1/c_t) for each event
print("\nCalculating uniqueness with calculate_average_uniqueness()...")
luna_avg_uniqueness = calculate_average_uniqueness(luna_ind_matrix)
ftx_avg_uniqueness = calculate_average_uniqueness(ftx_ind_matrix)

# Map back to timestamps and normalize
luna_weights = pd.Series(index=luna_timestamps.index, data=luna_avg_uniqueness.values)
ftx_weights = pd.Series(index=ftx_timestamps.index, data=ftx_avg_uniqueness.values)

# Normalize weights to sum to N (standard practice)
luna_weights = luna_weights * len(luna_weights) / luna_weights.sum()
ftx_weights = ftx_weights * len(ftx_weights) / ftx_weights.sum()

# Step 5: Get crisis period weights
luna_crisis_basis = data.loc[
    luna_crisis_start:luna_crisis_end, "luna_abnormal_basis"
].dropna()
ftx_crisis_basis = data.loc[
    ftx_crisis_start:ftx_crisis_end, "ftx_abnormal_basis"
].dropna()

luna_crisis_weights = luna_weights.reindex(luna_crisis_basis.index).dropna()
ftx_crisis_weights = ftx_weights.reindex(ftx_crisis_basis.index).dropna()

print("\n" + "=" * 60)
print("CRISIS PERIOD WEIGHTS:")
print(f"\nLUNA (N={len(luna_crisis_weights)}):")
print(f"  Mean: {luna_crisis_weights.mean():.3f}, Std: {luna_crisis_weights.std():.3f}")
print(f"  Range: [{luna_crisis_weights.min():.3f}, {luna_crisis_weights.max():.3f}]")

print(f"\nFTX (N={len(ftx_crisis_weights)}):")
print(f"  Mean: {ftx_crisis_weights.mean():.3f}, Std: {ftx_crisis_weights.std():.3f}")
print(f"  Range: [{ftx_crisis_weights.min():.3f}, {ftx_crisis_weights.max():.3f}]")


# Step 6: Compare weighted vs unweighted statistics
def weighted_mean(x, w):
    return np.average(x, weights=w)


def weighted_std(x, w):
    avg = weighted_mean(x, w)
    return np.sqrt(np.average((x - avg) ** 2, weights=w))


luna_aligned = luna_crisis_basis.loc[luna_crisis_weights.index]
ftx_aligned = ftx_crisis_basis.loc[ftx_crisis_weights.index]

print("\n" + "=" * 60)
print("WEIGHTED vs UNWEIGHTED STATISTICS")
print("=" * 60)
print("\nLUNA Abnormal Basis:")
print(f"  Unweighted: mean={luna_aligned.mean():.3f}%, std={luna_aligned.std():.3f}%")
print(
    f"  Weighted:   mean={weighted_mean(luna_aligned, luna_crisis_weights):.3f}%, std={weighted_std(luna_aligned, luna_crisis_weights):.3f}%"
)

print("\nFTX Abnormal Basis:")
print(f"  Unweighted: mean={ftx_aligned.mean():.3f}%, std={ftx_aligned.std():.3f}%")
print(
    f"  Weighted:   mean={weighted_mean(ftx_aligned, ftx_crisis_weights):.3f}%, std={weighted_std(ftx_aligned, ftx_crisis_weights):.3f}%"
)

# Appendix: FFD stationarity diagnostics and uniqueness weights.
stationarity_data = {
    "Metric": [
        "Optimal d",
        "Correlation (Memory)",
        "ADF Statistic",
        "Critical Value (5%)",
    ],
    "LUNA Series": [
        f"{luna_optimal_d:.2f}",
        f"{luna_corr_at_opt:.3f}",
        f"{luna_ffd_results.loc[luna_optimal_d, 'adfStat']:.4f}",
        f"{luna_ffd_results.loc[luna_optimal_d, '95% conf']:.4f}",
    ],
    "FTX Series": [
        f"{ftx_optimal_d:.2f}",
        f"{ftx_corr_at_opt:.3f}",
        f"{ftx_ffd_results.loc[ftx_optimal_d, 'adfStat']:.4f}",
        f"{ftx_ffd_results.loc[ftx_optimal_d, '95% conf']:.4f}",
    ],
}

# Optional: Add simple Pass/Fail check
is_luna_stat = (
    luna_ffd_results.loc[luna_optimal_d, "adfStat"]
    < luna_ffd_results.loc[luna_optimal_d, "95% conf"]
)
is_ftx_stat = (
    ftx_ffd_results.loc[ftx_optimal_d, "adfStat"]
    < ftx_ffd_results.loc[ftx_optimal_d, "95% conf"]
)

stationarity_data["LUNA Series"].append("Pass" if is_luna_stat else "Fail")
stationarity_data["FTX Series"].append("Pass" if is_ftx_stat else "Fail")
stationarity_data["Metric"].append("Stationarity Test")

tab_stationarity = pd.DataFrame(stationarity_data).set_index("Metric")
save_paper_table(tab_stationarity, "tab_appendix_stationarity_diagnostics")

# 2. Plot Uniqueness Weights Histogram (Methodology Justification)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    luna_crisis_weights,
    bins=30,
    alpha=0.6,
    color="#e74c3c",
    label="LUNA Weights",
    density=True,
)
ax.hist(
    ftx_crisis_weights,
    bins=30,
    alpha=0.6,
    color="#9b59b6",
    label="FTX Weights",
    density=True,
)

ax.set_title(
    "Distribution of Sample Uniqueness Weights", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Sample Weight (Uniqueness)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(
    0.05,
    0.95,
    "Note: Left-skewed weights indicate\nhigh data overlap (Redundancy)",
    transform=ax.transAxes,
    va="top",
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)

save_paper_fig("fig_appendix_weights_distribution")
plt.show()
