import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ==========================================
# 1. Dataset Generation
# ==========================================
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = 5000 * X + 20000 + np.random.normal(0, 3000, len(X))

# --- THE FIX: Feature Scaling (Standardization) ---
# This transforms X to have a mean of 0 and a standard deviation of 1.
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std

# ==========================================
# 2. Gradient Descent Algorithm
# ==========================================
def linear_regression_gd(X_data, y_data, learning_rate=0.1, epochs=100):
    n = len(X_data)
    m = 0.0  
    c = 0.0  
    cost_history = []

    for i in range(epochs):
        y_pred = m * X_data + c
        
        cost = (1 / (2 * n)) * np.sum((y_pred - y_data) ** 2)
        cost_history.append(cost)
        
        dm = (1 / n) * np.sum((y_pred - y_data) * X_data)
        dc = (1 / n) * np.sum(y_pred - y_data)
        
        m -= learning_rate * dm
        c -= learning_rate * dc
        
    return m, c, cost_history

# ==========================================
# 3. Train the Model
# ==========================================
# Notice we now need far fewer epochs (100) and can use a higher learning rate (0.1)
m_opt, c_opt, cost_history = linear_regression_gd(X_scaled, y, learning_rate=0.1, epochs=100)

# ==========================================
# 4. Plotting the Results
# ==========================================
plt.figure(figsize=(14, 5))

# Plot 1: The Regression Line vs Actual Data
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Actual Salaries')

# We calculate the line using the scaled features, but plot it against the original X
plt.plot(X, m_opt * X_scaled + c_opt, color='red', linewidth=2, label='Regression Line')
plt.title('Corrected Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Gradient Descent Optimization Curve
plt.subplot(1, 2, 2)
plt.plot(range(100), cost_history, color='green', linewidth=2)
plt.title('Gradient Descent Cost Curve (Cost vs Epoch)')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Cost (Mean Squared Error)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "linear_regression.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()