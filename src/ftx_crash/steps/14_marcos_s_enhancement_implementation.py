"""Notebook section: marcos s enhancement implementation."""

# This step pulls in the modules used by enhancements 15-19. The names land in the
# shared execution namespace, so later step files use them without importing again.
import RiskLabAI.data.differentiation.differentiation as diff
import RiskLabAI.data.synthetic_data as synth
import statsmodels.api as sm
from RiskLabAI.data.labeling.labeling import (
    cusum_filter_events_dynamic_threshold,
    daily_volatility_with_log_returns,
    symmetric_cusum_filter,
)
from RiskLabAI.data.weights.sample_weights import (
    calculate_average_uniqueness,
    expand_label_for_meta_labeling,
)
