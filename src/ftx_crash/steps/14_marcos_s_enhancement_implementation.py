"""Notebook section: marcos s enhancement implementation."""

# RiskLabAI is imported in step 01; this step pulls in modules used by enhancements 15-19.
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
