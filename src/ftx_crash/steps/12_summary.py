"""Notebook section: summary."""

summary_data = {
    "Metric": [
        "Normal Basis",
        "Crisis Avg Abnormal Basis",
        "Crisis Volatility (Std)",
        "Half-Life (hours)",
        "Half-Life (days)",
    ],
    "Terra/LUNA": [
        f"{luna_normal_front_basis:.2f}%",
        f"{luna_avg_abnormal_basis:.2f}%",
        f"{luna_ab_basis.std():.2f}%",
        f"{luna_basis_half_life:.1f}",
        f"{luna_basis_half_life / 24:.1f}",
    ],
    "FTX": [
        f"{ftx_normal_front_basis:.2f}%",
        f"{ftx_avg_abnormal_basis:.2f}%",
        f"{ftx_ab_basis.std():.2f}%",
        f"{ftx_basis_half_life:.1f}",
        f"{ftx_basis_half_life / 24:.1f}",
    ],
    "Ratio (FTX/LUNA)": [
        f"{ftx_normal_front_basis / luna_normal_front_basis:.2f}x",
        f"{abs(ftx_avg_abnormal_basis / luna_avg_abnormal_basis):.2f}x",
        f"{ftx_ab_basis.std() / luna_ab_basis.std():.2f}x",
        f"{ftx_basis_half_life / luna_basis_half_life:.2f}x",
        f"{ftx_basis_half_life / luna_basis_half_life:.2f}x",
    ],
}

summary_df = pd.DataFrame(summary_data)
print("\n" + "=" * 80)
print("COMPARATIVE SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))
print("\n" + "=" * 80)

print("\nStatistical tests:")
print(
    f"  T-test:        t = {basis_ttest.statistic:7.3f}, p = {basis_ttest.pvalue:.2e}"
)
print(
    f"  Mann-Whitney:  U = {basis_utest.statistic:7.1f}, p = {basis_utest.pvalue:.2e}"
)
print(f"  Cohen's d:     d = {basis_cohens_d:7.3f} ({effect_size} effect)")
