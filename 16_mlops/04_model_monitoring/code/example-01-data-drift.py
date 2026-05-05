import numpy as np
from scipy import stats

def check_drift(baseline_data, live_data, feature_name):
    """
    Uses the Kolmogorov-Smirnov test to detect data drift.
    """
    # Perform the K-S test
    statistic, p_value = stats.ks_2samp(baseline_data, live_data)
    
    print(f"\n--- Testing Feature: {feature_name} ---")
    print(f"K-S Statistic: {statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Typical threshold is 0.05
    if p_value < 0.05:
        print("🚨 ALERT: Data Drift Detected! The distributions are significantly different.")
    else:
        print("✅ No significant drift detected.")

# 1. Simulate Baseline (Training) Data
# E.g., The age of our users when we trained the model
print("Gathering Baseline Data...")
np.random.seed(42)
baseline_age = np.random.normal(loc=35, scale=5, size=1000)

# 2. Simulate Live Data (Week 1)
# The user base looks basically the same
live_age_week1 = np.random.normal(loc=35.2, scale=5.1, size=1000)
check_drift(baseline_age, live_age_week1, "User Age (Week 1)")

# 3. Simulate Live Data (Month 6)
# Our marketing campaign brought in a much younger demographic
live_age_month6 = np.random.normal(loc=28, scale=6, size=1000)
check_drift(baseline_age, live_age_month6, "User Age (Month 6)")

# What does this mean?
# Our model was trained to predict behavior for 35-year-olds.
# It is now mostly seeing 28-year-olds. It might not be accurate anymore.
