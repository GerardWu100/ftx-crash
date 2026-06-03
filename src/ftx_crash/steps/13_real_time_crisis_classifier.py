"""Notebook section: real time crisis classifier."""

# Feature Engineering for Crisis Classification
# Extract features from the first N hours of each crisis

def extract_crisis_features(crisis_data, basis_col, slope_col, n_bars=6):
    """Extract early-warning features from the first ``n_bars`` 1-minute observations.

    Note: ``n_bars=6`` is six minutes on 1-min data (notebook name ``n_hours`` was misleading).
    """
    early_data = crisis_data.iloc[:n_bars].copy()

    basis_series = early_data[basis_col].dropna()
    slope_series = early_data[slope_col].dropna() if slope_col in early_data.columns else pd.Series([0])

    features = {
        'mean_basis_dislocation': basis_series.mean() if len(basis_series) > 0 else 0,
        'vol_of_basis': basis_series.std() if len(basis_series) > 1 else 0,
        'autocorr_basis': basis_series.autocorr(lag=1) if len(basis_series) > 2 else 0,
        'mean_term_structure_slope': slope_series.mean() if len(slope_series) > 0 else 0,
    }

    # Handle NaN autocorrelation
    if pd.isna(features['autocorr_basis']):
        features['autocorr_basis'] = 0

    return features

# Extract features for LUNA and FTX crises
luna_features = extract_crisis_features(
    data.loc[luna_crisis_start:luna_crisis_end],
    'luna_abnormal_basis', 'luna_abnormal_slope', n_bars=6
)

ftx_features = extract_crisis_features(
    data.loc[ftx_crisis_start:ftx_crisis_end],
    'ftx_abnormal_basis', 'ftx_abnormal_slope', n_bars=6
)

print("Early Crisis Features (First 6 bars / minutes):")
print("=" * 60)
print(f"\n{'Feature':<30} {'LUNA (Protocol)':<15} {'FTX (Counterparty)':<15}")
print("-" * 60)
for feature in luna_features.keys():
    luna_val = luna_features[feature]
    ftx_val = ftx_features[feature]
    print(f"{feature:<30} {luna_val:>12.4f} {ftx_val:>17.4f}")

print("\nNote: Positive autocorrelation indicates persistent dislocations")

# Generate synthetic crisis events using bootstrap resampling
# Since we only have 2 real events, we create synthetic variations

def generate_synthetic_events(base_features, label, n_samples=50, noise_scale=0.3, seed=42):
    """
    Generate synthetic crisis events by adding noise to base features.

    This simulates what we might observe from similar crisis types
    with natural variation in market conditions.
    """
    np.random.seed(seed)

    samples = []
    feature_names = list(base_features.keys())

    for i in range(n_samples):
        sample = {}
        for feat in feature_names:
            base_val = base_features[feat]
            # Add Gaussian noise proportional to the feature magnitude
            noise = np.random.normal(0, abs(base_val) * noise_scale + 0.01)
            sample[feat] = base_val + noise
        sample['crisis_type'] = label
        samples.append(sample)

    return pd.DataFrame(samples)

# Generate synthetic datasets
# Label 0 = Protocol Crisis (LUNA-like)
# Label 1 = Counterparty Crisis (FTX-like)
n_synthetic = 100

luna_synthetic = generate_synthetic_events(luna_features, label=0, n_samples=n_synthetic, seed=42)
ftx_synthetic = generate_synthetic_events(ftx_features, label=1, n_samples=n_synthetic, seed=123)

# Combine into training dataset
crisis_df = pd.concat([luna_synthetic, ftx_synthetic], ignore_index=True)
crisis_df = crisis_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"Synthetic Dataset Created:")
print(f"  Protocol (LUNA-like) events: {(crisis_df['crisis_type'] == 0).sum()}")
print(f"  Counterparty (FTX-like) events: {(crisis_df['crisis_type'] == 1).sum()}")
print(f"  Total samples: {len(crisis_df)}")

print("\nFeature distributions by crisis type:")
print(crisis_df.groupby('crisis_type').mean().T.round(4))

# Train Crisis Classifiers
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Prepare features and target
feature_cols = ['mean_basis_dislocation', 'vol_of_basis', 'autocorr_basis', 'mean_term_structure_slope']
X = crisis_df[feature_cols]
y = crisis_df['crisis_type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Cross-validation scores
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

# Predictions
y_pred = rf_model.predict(X_test)

print("Random Forest Classifier Results:")
print("=" * 60)
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Protocol (LUNA)', 'Counterparty (FTX)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"                 Protocol  Counterparty")
print(f"Actual Protocol      {cm[0,0]:3d}         {cm[0,1]:3d}")
print(f"      Counterparty   {cm[1,0]:3d}         {cm[1,1]:3d}")

# ---------------------------------------------------------
# FEATURE IMPORTANCE ANALYSIS (Standard Sklearn)
# ---------------------------------------------------------
# Replaces SHAP to avoid dependency errors.
# Uses Gini Impurity to measure feature power.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get feature importances directly from the trained Random Forest model
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances,
    'std': std
}).sort_values('importance', ascending=False)

print("Random Forest Feature Importance (Mean Decrease in Impurity):")
print("=" * 60)
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f} (+/- {row['std']*2:.4f})")

# Identify top predictor
top_feature = feature_importance.iloc[0]['feature']
print(f"\nTop Predictor: {top_feature}")

if 'autocorr' in top_feature.lower():
    print("\nHypothesis CONFIRMED: Autocorrelation (persistence) is the #1 predictor!")
    print("This supports the theory that counterparty crises exhibit a unique")
    print("'fingerprint' of sustained arbitrage breakdown detectable early on.")
else:
    print(f"\nNote: {top_feature} is the strongest predictor in this analysis.")



# Save Table
print("\n--- Saving Feature Importance Table ---")
save_table(feature_importance, 'tab4_feature_importance')

# Visualization & Save Figure
plt.figure(figsize=(10, 6))
colors = ['#E74C3C' if 'autocorr' in f else '#3498DB' for f in feature_importance['feature']]
bars = plt.barh(feature_importance['feature'], feature_importance['importance'], 
         xerr=feature_importance['std'], capsize=5, color=colors, alpha=0.8)
plt.xlabel('Mean Decrease in Impurity', fontsize=12)
plt.title('Crisis Classification: Top Predictors', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis() 
plt.grid(axis='x', alpha=0.3)

# Add values
for bar, val in zip(bars, feature_importance['importance']):
    plt.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_feature_importance.png'), dpi=PLOT_QUALITY, bbox_inches='tight')
plt.show()
