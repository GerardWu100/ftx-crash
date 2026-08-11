"""Notebook section: crisis comparison."""

# Create comprehensive crisis comparison visualization
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
plt.style.use("seaborn-v0_8-darkgrid")

# ========== PLOT 1: Time-Aligned Crisis Trajectory ==========
ax1 = fig.add_subplot(gs[0, :2])
luna_aligned = data.loc[luna_crisis_start:luna_post_end, "luna_abnormal_basis"].copy()
ftx_aligned = data.loc[ftx_crisis_start:ftx_post_end, "ftx_abnormal_basis"].copy()

luna_hours = [
    (t - luna_crisis_start).total_seconds() / 3600 for t in luna_aligned.index
]
ftx_hours = [(t - ftx_crisis_start).total_seconds() / 3600 for t in ftx_aligned.index]

ax1.plot(
    luna_hours,
    luna_aligned.values,
    label="Terra/LUNA (Protocol)",
    linewidth=3,
    color="#e74c3c",
    alpha=0.85,
)
ax1.plot(
    ftx_hours,
    ftx_aligned.values,
    label="FTX (Counterparty)",
    linewidth=3,
    color="#9b59b6",
    alpha=0.85,
)

luna_crisis_hours = (luna_crisis_end - luna_crisis_start).total_seconds() / 3600
ax1.axvspan(0, luna_crisis_hours, alpha=0.12, color="red", label="Crisis Window")
ax1.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)

ax1.set_title(
    "Crisis Evolution: Time-Aligned Abnormal Basis",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax1.set_xlabel("Hours Since Crisis Start", fontsize=13)
ax1.set_ylabel("Abnormal Basis (%)", fontsize=13)
ax1.legend(loc="lower right", fontsize=12, framealpha=0.95)
ax1.grid(True, alpha=0.35)

# Add annotations for key events
ax1.annotate(
    "FTX peak\ndislocation",
    xy=(120, ftx_aligned.min()),
    xytext=(180, ftx_aligned.min() + 2),
    fontsize=10,
    ha="left",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#9b59b6", alpha=0.3),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="#9b59b6"),
)

# ========== PLOT 2: Statistical Distribution ==========
ax2 = fig.add_subplot(gs[0, 2])
violin_data = [luna_ab_basis.dropna(), ftx_ab_basis.dropna()]
parts = ax2.violinplot(violin_data, positions=[1, 2], showmeans=True, showmedians=True)

for pc, color in zip(parts["bodies"], ["#e74c3c", "#9b59b6"]):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(["LUNA", "FTX"])
ax2.set_ylabel("Abnormal Basis (%)", fontsize=12)
ax2.set_title("Distribution Comparison", fontsize=13, fontweight="bold", pad=12)
ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax2.grid(True, alpha=0.3, axis="y")

# Add mean values
for i, (data_series, x_pos) in enumerate(zip(violin_data, [1, 2])):
    mean_val = data_series.mean()
    ax2.text(
        x_pos,
        mean_val,
        f"{mean_val:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

# ========== PLOT 3: Three Dimensions Bar Chart ==========
ax3 = fig.add_subplot(gs[1, :])
metrics = [
    "Magnitude\n|Avg Abnormal Basis|",
    "Volatility\n(Std Dev)",
    "Persistence\n(Half-Life, hours)",
]
luna_vals = [abs(luna_avg_abnormal_basis), luna_ab_basis.std(), luna_basis_half_life]
ftx_vals = [abs(ftx_avg_abnormal_basis), ftx_ab_basis.std(), ftx_basis_half_life]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(
    x - width / 2,
    luna_vals,
    width,
    label="Terra/LUNA",
    color="#e74c3c",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)
bars2 = ax3.bar(
    x + width / 2,
    ftx_vals,
    width,
    label="FTX",
    color="#9b59b6",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)

ax3.set_xlabel("Crisis Impact Dimension", fontsize=13, fontweight="bold")
ax3.set_ylabel("Value", fontsize=12)
ax3.set_title(
    "Three-Dimensional Crisis Comparison", fontsize=15, fontweight="bold", pad=15
)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.legend(loc="upper left", fontsize=12)
ax3.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

# Add ratio annotations above each group
ratios = [
    abs(ftx_avg_abnormal_basis / luna_avg_abnormal_basis),
    ftx_ab_basis.std() / luna_ab_basis.std(),
    ftx_basis_half_life / luna_basis_half_life,
]

for i, ratio in enumerate(ratios):
    ax3.text(
        i,
        max(luna_vals[i], ftx_vals[i]) * 1.15,
        f"FTX/LUNA\n{ratio:.1f}x",
        ha="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e67e22", alpha=0.6),
    )

plt.tight_layout()
# ... (existing multi-panel plotting code) ...
plt.savefig(
    os.path.join(FIG_DIR, "fig2_crisis_comparison_panels.png"),
    dpi=PLOT_QUALITY,
    bbox_inches="tight",
)
plt.show()

# Print quantitative summary
print("\n" + "=" * 80)
print("CRISIS IMPACT ANALYSIS: QUANTITATIVE RESULTS")
print("=" * 80)
print(f"\n{'Metric':<25} {'Terra/LUNA':<15} {'FTX':<15} {'Ratio (FTX/LUNA)':<15}")
print("-" * 80)
print(
    f"{'Magnitude (|avg|)':<25} {abs(luna_avg_abnormal_basis):>8.2f}%      {abs(ftx_avg_abnormal_basis):>8.2f}%      {abs(ftx_avg_abnormal_basis / luna_avg_abnormal_basis):>8.1f}x"
)
print(
    f"{'Volatility (std)':<25} {luna_ab_basis.std():>8.2f}%      {ftx_ab_basis.std():>8.2f}%      {ftx_ab_basis.std() / luna_ab_basis.std():>8.1f}x"
)
print(
    f"{'Persistence (hours)':<25} {luna_basis_half_life:>8.1f}h       {ftx_basis_half_life:>8.1f}h       {ftx_basis_half_life / luna_basis_half_life:>8.1f}x"
)
print(
    f"{'Persistence (days)':<25} {luna_basis_half_life / 24:>8.1f}d       {ftx_basis_half_life / 24:>8.1f}d       {ftx_basis_half_life / luna_basis_half_life:>8.1f}x"
)
print("=" * 80)
