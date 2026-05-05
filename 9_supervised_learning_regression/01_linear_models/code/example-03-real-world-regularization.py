import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate a dataset with some "useless" or highly correlated features (multicollinearity)
np.random.seed(42)
n_samples = 500

# True useful features
house_size = np.random.uniform(800, 4000, n_samples)
num_rooms = np.round(house_size / 400 + np.random.normal(0, 0.5, n_samples))
age = np.random.uniform(0, 50, n_samples)

# Redundant/Noisy features
size_in_cm = house_size * 929.03  # Perfectly correlated with house_size
random_noise_1 = np.random.normal(0, 100, n_samples)
random_noise_2 = np.random.normal(0, 50, n_samples)

# Target: Price
price = 50000 + (150 * house_size) - (1000 * age) + np.random.normal(0, 20000, n_samples)

df = pd.DataFrame({
    'House_Size_sqft': house_size,
    'Num_Rooms': num_rooms,
    'Age_Years': age,
    'Size_cm2': size_in_cm,
    'Noise_1': random_noise_1,
    'Noise_2': random_noise_2
})

X = df
y = price

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CRITICAL STEP: Scale the data before Regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models
# A. Standard OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# B. Ridge (L2 Penalty)
ridge = Ridge(alpha=100.0) # alpha is regularization strength
ridge.fit(X_train_scaled, y_train)

# C. Lasso (L1 Penalty) - Good for feature selection
lasso = Lasso(alpha=1000.0)
lasso.fit(X_train_scaled, y_train)

# 5. Evaluate and Compare
models = {'OLS': ols, 'Ridge': ridge, 'Lasso': lasso}

for name, model in models.items():
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    print(f"\n--- {name} Model ---")
    print(f"Test R-squared: {r2:.4f}")
    
    # Print coefficients to see the effect of regularization
    print("Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.2f}")

print("\n--- Summary of Regularization Effects ---")
print("1. OLS struggles because 'House_Size_sqft' and 'Size_cm2' are perfectly correlated. It gives wild, arbitrary weights to them.")
print("2. Ridge distributes the weight somewhat evenly between correlated features and shrinks noise.")
print("3. Lasso completely drops 'Size_cm2' and the noise columns (coefficients = 0.00), performing automatic feature selection!")
