"""Notebook section: crisis impact metrics."""

def calculate_half_life(series):
    """Calculate half-life of mean reversion using AR(1) model.

    For an AR(1) process: X_t = rho * X_{t-1} + epsilon_t
    Half-life = -ln(2) / ln(rho)

    Note: This assumes the series is already demeaned (abnormal basis).
    For non-zero mean processes, use OLS with intercept.
    """
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return np.nan

    y = clean_series.values[1:]
    y_lag = clean_series.values[:-1]

    # OLS estimate of rho (no intercept since abnormal basis is demeaned)
    rho = np.sum(y * y_lag) / np.sum(y_lag ** 2)

    # Handle edge cases
    if rho >= 1:  # Non-stationary (unit root or explosive)
        return np.inf
    elif rho <= 0:  # Negative autocorrelation (oscillatory)
        return np.inf  # Half-life undefined for negative rho

    return -np.log(2) / np.log(rho)

# Calculate cumulative and average abnormal basis during crisis windows
luna_crisis_data = data.loc[luna_crisis_start:luna_crisis_end]
luna_cum_abnormal_basis = luna_crisis_data['luna_abnormal_basis'].sum()
luna_avg_abnormal_basis = luna_crisis_data['luna_abnormal_basis'].mean()

ftx_crisis_data = data.loc[ftx_crisis_start:ftx_crisis_end]
ftx_cum_abnormal_basis = ftx_crisis_data['ftx_abnormal_basis'].sum()
ftx_avg_abnormal_basis = ftx_crisis_data['ftx_abnormal_basis'].mean()

# Calculate half-life for persistence analysis (using post-crisis recovery period)
luna_post_ab_basis = data.loc[luna_post_start:luna_post_end, 'luna_abnormal_basis']
ftx_post_ab_basis = data.loc[ftx_post_start:ftx_post_end, 'ftx_abnormal_basis']

luna_basis_half_life = calculate_half_life(luna_post_ab_basis)
ftx_basis_half_life = calculate_half_life(ftx_post_ab_basis)

# Extract crisis period abnormal basis for statistical tests
luna_ab_basis = luna_crisis_data['luna_abnormal_basis'].dropna()
ftx_ab_basis = ftx_crisis_data['ftx_abnormal_basis'].dropna()
luna_ab_slope = luna_crisis_data['luna_abnormal_slope'].dropna()
ftx_ab_slope = ftx_crisis_data['ftx_abnormal_slope'].dropna()

print("Crisis Period Summary:")
print("\nTerra/LUNA:")
print(f"  Average abnormal basis: {luna_avg_abnormal_basis:7.2f}%")
print(f"  Cumulative impact:      {luna_cum_abnormal_basis:7.2f}%")
print(f"  Half-life (recovery):   {luna_basis_half_life:5.1f}h ({luna_basis_half_life/24:.1f}d)")

print("\nFTX:")
print(f"  Average abnormal basis: {ftx_avg_abnormal_basis:7.2f}%")
print(f"  Cumulative impact:      {ftx_cum_abnormal_basis:7.2f}%")
print(f"  Half-life (recovery):   {ftx_basis_half_life:5.1f}h ({ftx_basis_half_life/24:.1f}d)")
