"""Notebook section: statistical tests."""

# Statistical tests comparing abnormal basis between crises
basis_ttest = stats.ttest_ind(luna_ab_basis, ftx_ab_basis, equal_var=False)

# Mann-Whitney U test (non-parametric alternative)
basis_utest = stats.mannwhitneyu(luna_ab_basis, ftx_ab_basis, alternative='two-sided')

# Effect size (Cohen's d)
def cohens_d(x1, x2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std

basis_cohens_d = cohens_d(ftx_ab_basis, luna_ab_basis)

# Determine effect size category based on Cohen's d
def get_effect_size_category(d):
    """Return effect size category based on Cohen's d value."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"

effect_size = get_effect_size_category(basis_cohens_d)

# Difference-in-Differences estimator
did_estimate = ftx_avg_abnormal_basis - luna_avg_abnormal_basis

print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

print(f"\nAbnormal basis:")
print(f"  Terra/LUNA: {luna_ab_basis.mean():7.2f}% (std: {luna_ab_basis.std():.2f}%)")
print(f"  FTX:        {ftx_ab_basis.mean():7.2f}% (std: {ftx_ab_basis.std():.2f}%)")
print(f"  Difference: {ftx_ab_basis.mean() - luna_ab_basis.mean():7.2f}%")

print(f"\nTests:")
print(f"  T-test:        t = {basis_ttest.statistic:7.3f}, p = {basis_ttest.pvalue:.2e}")
print(f"  Mann-Whitney:  U = {basis_utest.statistic:7.1f}, p = {basis_utest.pvalue:.2e}")
print(f"  Cohen's d:     d = {basis_cohens_d:7.3f} ({effect_size} effect)")

if basis_ttest.pvalue < 0.001:
    print(f"\nResult: Highly significant difference (p < 0.001)")
elif basis_ttest.pvalue < 0.05:
    print(f"\nResult: Significant difference (p < 0.05)")
else:
    print(f"\nResult: No significant difference (p >= 0.05)")

print("=" * 70)


def significance_stars(p_value: float) -> str:
    """Map p-value to star notation for summary tables."""
    if p_value < 0.001:
        return '***'
    if p_value < 0.05:
        return '**'
    return 'NS'


# Exportable table of formal tests on abnormal basis during crisis windows.
stat_results = pd.DataFrame([
    {
        'Test': "Welch's t-test", 
        'Statistic': basis_ttest.statistic, 
        'P-Value': basis_ttest.pvalue, 
        'Significance': significance_stars(basis_ttest.pvalue),
    },
    {
        'Test': "Mann-Whitney U", 
        'Statistic': basis_utest.statistic, 
        'P-Value': basis_utest.pvalue, 
        'Significance': significance_stars(basis_utest.pvalue),
    },
    {
        'Test': "Cohen's d (Effect Size)", 
        'Statistic': basis_cohens_d, 
        'P-Value': np.nan, 
        'Significance': get_effect_size_category(basis_cohens_d)
    }
]).set_index('Test')

print("\n--- Saving Statistical Test Results ---")
save_table(stat_results, 'tab2_statistical_tests')
