"""Notebook section: event study abnormal basis."""

# Normal (estimation-period) basis is the benchmark for abnormal basis during crises.
luna_est_data = data.loc[luna_estimation_start:luna_estimation_end]
luna_normal_front_basis = luna_est_data["luna_front_basis"].mean()

ftx_est_data = data.loc[ftx_estimation_start:ftx_estimation_end]
ftx_normal_front_basis = ftx_est_data["ftx_front_basis"].mean()

# Abnormal basis = realized basis minus its pre-crisis mean (standard event-study residual).
data["luna_abnormal_basis"] = data["luna_front_basis"] - luna_normal_front_basis
data["ftx_abnormal_basis"] = data["ftx_front_basis"] - ftx_normal_front_basis

# Term-structure slope abnormal values (FTX slope is NaN when only BITO proxy is available).
luna_normal_slope = luna_est_data["luna_slope"].mean()
ftx_normal_slope = ftx_est_data["ftx_slope"].mean()

data["luna_abnormal_slope"] = data["luna_slope"] - luna_normal_slope
data["ftx_abnormal_slope"] = data["ftx_slope"] - ftx_normal_slope

print("\nNormal basis (estimation period):")
print(f"  Terra/LUNA: {luna_normal_front_basis:6.2f}%")
print(f"  FTX:        {ftx_normal_front_basis:6.2f}%")
