import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 1. Generate Non-Linear Data (Sine wave with noise)
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(16))

# 2. CRITICAL: SVR Requires Data Scaling!
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 3. Train SVR Models with different Kernels
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1, coef0=1)
svr_lin = SVR(kernel='linear', C=100)

svr_rbf.fit(X_scaled, y_scaled)
svr_poly.fit(X_scaled, y_scaled)
svr_lin.fit(X_scaled, y_scaled)

# 4. Predict
# We use the scaled X to predict, then inverse transform to get the real y values back
y_rbf = scaler_y.inverse_transform(svr_rbf.predict(X_scaled).reshape(-1, 1)).ravel()
y_poly = scaler_y.inverse_transform(svr_poly.predict(X_scaled).reshape(-1, 1)).ravel()
y_lin = scaler_y.inverse_transform(svr_lin.predict(X_scaled).reshape(-1, 1)).ravel()

# 5. Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data')
plt.plot(X, y_rbf, color='red', lw=2, label='RBF model (Non-linear)')
plt.plot(X, y_poly, color='blue', lw=2, label='Polynomial model (degree 3)')
plt.plot(X, y_lin, color='green', lw=2, label='Linear model')

plt.title('Support Vector Regression (SVR)')
plt.xlabel('X data')
plt.ylabel('y value')
plt.legend()
plt.show()

print("Notice how the RBF kernel bends to perfectly fit the sine wave shape, while the linear kernel fails completely.")
