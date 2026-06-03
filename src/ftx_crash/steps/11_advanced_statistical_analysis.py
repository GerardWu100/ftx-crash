"""Notebook section: advanced statistical analysis."""

# Enhanced statistical analysis visualization
from scipy.stats import t as t_dist

fig, ax1 = plt.subplots(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# ========== Statistical Test Results ==========
# Prepare test results data
test_names = ['Welch\'s t-test', 'Mann-Whitney U', 'Cohen\'s d']
test_values = [
    -np.log10(basis_ttest.pvalue),  # Convert p-value to -log10 scale
    -np.log10(basis_utest.pvalue),
    abs(basis_cohens_d)
]
test_labels = [
    f'p = {basis_ttest.pvalue:.2e}',
    f'p = {basis_utest.pvalue:.2e}',
    f'd = {basis_cohens_d:.3f}'
]

colors_tests = ['#3498db', '#2ecc71', '#e67e22']
bars = ax1.barh(test_names, test_values, color=colors_tests, alpha=0.8,
                edgecolor='black', linewidth=1.5)

# Add significance threshold lines
ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2,
           label='p = 0.05 threshold', alpha=0.7)
ax1.axvline(x=-np.log10(0.001), color='darkred', linestyle='--', linewidth=2,
           label='p = 0.001 threshold', alpha=0.7)

ax1.set_xlabel('-log10(p-value) | Effect Size', fontsize=12, fontweight='bold')
ax1.set_title('Statistical Significance Tests', fontsize=14, fontweight='bold', pad=12)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for bar, label in zip(bars, test_labels):
    width = bar.get_width()
    ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
            label, ha='left', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
save_paper_fig('fig_statistical_significance')
plt.show()

# Calculate 95% confidence intervals for text output
def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean = data.mean()
    se = data.std() / np.sqrt(n)
    margin = t_dist.ppf((1 + confidence) / 2, n - 1) * se
    return mean, mean - margin, mean + margin

luna_mean, luna_ci_lower, luna_ci_upper = calculate_ci(luna_ab_basis)
ftx_mean, ftx_ci_lower, ftx_ci_upper = calculate_ci(ftx_ab_basis)

# Detailed statistical output
print("\n" + "="*80)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("="*80)

print("\n1. DESCRIPTIVE STATISTICS")
print("-"*80)
print(f"{'Crisis':<15} {'Mean':<12} {'Median':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
print("-"*80)
print(f"{'Terra/LUNA':<15} {luna_ab_basis.mean():>8.2f}%   {luna_ab_basis.median():>8.2f}%   "
      f"{luna_ab_basis.std():>8.2f}%   {luna_ab_basis.min():>8.2f}%   {luna_ab_basis.max():>8.2f}%")
print(f"{'FTX':<15} {ftx_ab_basis.mean():>8.2f}%   {ftx_ab_basis.median():>8.2f}%   "
      f"{ftx_ab_basis.std():>8.2f}%   {ftx_ab_basis.min():>8.2f}%   {ftx_ab_basis.max():>8.2f}%")

print("\n2. CONFIDENCE INTERVALS (95%)")
print("-"*80)
print(f"Terra/LUNA: [{luna_ci_lower:.2f}%, {luna_ci_upper:.2f}%]  (width: {luna_ci_upper-luna_ci_lower:.2f}%)")
print(f"FTX:        [{ftx_ci_lower:.2f}%, {ftx_ci_upper:.2f}%]  (width: {ftx_ci_upper-ftx_ci_lower:.2f}%)")

print("\n3. HYPOTHESIS TESTS")
print("-"*80)
print(f"H0: mu_FTX = mu_LUNA (no difference)")
print(f"H1: mu_FTX != mu_LUNA (significant difference)")
print(f"\n  a) Welch's t-test (parametric):")
print(f"     Statistic: t = {basis_ttest.statistic:.3f}")
print(f"     P-value:   p = {basis_ttest.pvalue:.2e}")
print(f"     Result:    {'REJECT H0' if basis_ttest.pvalue < 0.05 else 'FAIL TO REJECT H0'} at alpha=0.05")

print(f"\n  b) Mann-Whitney U test (non-parametric):")
print(f"     Statistic: U = {basis_utest.statistic:.1f}")
print(f"     P-value:   p = {basis_utest.pvalue:.2e}")
print(f"     Result:    {'REJECT H0' if basis_utest.pvalue < 0.05 else 'FAIL TO REJECT H0'} at alpha=0.05")

print(f"\n  c) Effect Size (Cohen's d):")
print(f"     Cohen's d = {basis_cohens_d:.3f}")
effect_interpretation = 'large' if abs(basis_cohens_d) >= 0.8 else ('medium' if abs(basis_cohens_d) >= 0.5 else 'small')
print(f"     Interpretation: {effect_interpretation} effect size")

print("\n4. DIFFERENCE-IN-DIFFERENCES ESTIMATOR")
print("-"*80)
print(f"delta_DiD = mu_FTX - mu_LUNA = {did_estimate:.2f}%")
print(f"Interpretation: FTX caused {abs(did_estimate):.2f}% additional basis disruption")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if basis_ttest.pvalue < 0.001:
    print("Highly significant difference between crises (p < 0.001)")
elif basis_ttest.pvalue < 0.05:
    print("Significant difference between crises (p < 0.05)")
else:
    print("No statistically significant difference (p >= 0.05)")

print(f"Both parametric and non-parametric tests confirm the finding.")
print(f"The effect size is {effect_interpretation}, indicating {'strong' if effect_interpretation=='large' else 'moderate'} practical significance.")
print("="*80)
