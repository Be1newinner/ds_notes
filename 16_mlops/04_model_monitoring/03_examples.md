# Examples: Model Monitoring

This document outlines the practical examples provided for learning model monitoring.

## Code References

- `code/example-01-data-drift.py` — **Detecting Data Drift**: Simulates a baseline training dataset and a "live" dataset where the inputs have slowly changed. Uses `scipy.stats` to perform a Kolmogorov-Smirnov test to detect if the drift is statistically significant.
- `code/example-02-concept-drift.py` — **Simulating Concept Drift**: Trains a simple model. Simulates a scenario where the relationship between inputs and outputs changes (the "rules" change), but the inputs look identical. Shows how the model's accuracy slowly decays over time.

## How to use these examples

Run the Python scripts from your terminal. 
- Example 01 demonstrates the mathematical tests running behind the scenes of popular monitoring dashboards.
- Example 02 is purely illustrative to help visualize *why* models fail silently.
