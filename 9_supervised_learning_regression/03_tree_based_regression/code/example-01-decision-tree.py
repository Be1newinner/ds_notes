import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# 1. Generate Simple Data (Experience vs Salary)
np.random.seed(42)
experience = np.sort(np.random.uniform(1, 10, 20)).reshape(-1, 1)
# Salary generally goes up with experience, but in steps
salary = np.where(experience < 3, 40000, 
                  np.where(experience < 7, 70000, 100000)).ravel()
salary = salary + np.random.normal(0, 5000, 20) # add noise

# 2. Train a Decision Tree
# A depth of 2 means it can only ask 2 levels of questions
tree_model = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_model.fit(experience, salary)

# 3. Predict to visualize the "step" function
X_plot = np.arange(0, 11, 0.1).reshape(-1, 1)
y_plot = tree_model.predict(X_plot)

# 4. Visualize the Predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(experience, salary, color='black', label='Data')
plt.plot(X_plot, y_plot, color='red', lw=2, label='Tree Prediction')
plt.title('Decision Tree: Step Function Predictions')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

# 5. Visualize the Actual Tree Logic
plt.subplot(1, 2, 2)
plot_tree(tree_model, feature_names=['Experience'], filled=True, rounded=True)
plt.title('The Decision Tree Rules')

plt.tight_layout()
plt.show()

print("Notice how the Decision Tree doesn't draw a smooth line.")
print("Instead, it breaks the data into chunks (e.g., Exp < 2.5) and predicts the average salary for that chunk.")
