import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Binary Logistic Regression
# ==========================================
class BinaryLogisticRegression:
    def __init__(self, learning_rate=0.05, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.cost_history = []
        
    def sigmoid(self, z):
        # Clip z to prevent overflow in exp
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.b = 0
        
        # Ensure y is a column vector
        y = y.reshape(-1, 1)
        
        for i in range(self.epochs):
            # 1. Linear combination & Activation
            Z = np.dot(X, self.W) + self.b
            A = self.sigmoid(Z)
            
            # 2. Cost Calculation (Binary Cross-Entropy)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            cost = -(1/m) * np.sum(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))
            self.cost_history.append(cost)
            
            # 3. Gradients
            dW = (1/m) * np.dot(X.T, (A - y))
            db = (1/m) * np.sum(A - y)
            
            # 4. Update parameters
            self.W -= self.lr * dW
            self.b -= self.lr * db
            
    def predict(self, X):
        Z = np.dot(X, self.W) + self.b
        return (self.sigmoid(Z) >= 0.5).astype(int)

# ==========================================
# 2. Multiclass Logistic Regression (Softmax)
# ==========================================
class MulticlassLogisticRegression:
    def __init__(self, learning_rate=0.05, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.cost_history = []
        
    def softmax(self, z):
        # Shift z for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def to_one_hot(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def fit(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))
        y_oh = self.to_one_hot(y, num_classes)
        
        # W shape: (n_features, n_classes), b shape: (1, n_classes)
        self.W = np.zeros((n, num_classes))
        self.b = np.zeros((1, num_classes))
        
        for i in range(self.epochs):
            # 1. Linear combination & Softmax Activation
            Z = np.dot(X, self.W) + self.b
            A = self.softmax(Z)
            
            # 2. Cost Calculation (Categorical Cross-Entropy)
            epsilon = 1e-15
            cost = -(1/m) * np.sum(y_oh * np.log(A + epsilon))
            self.cost_history.append(cost)
            
            # 3. Gradients
            dW = (1/m) * np.dot(X.T, (A - y_oh))
            db = (1/m) * np.sum(A - y_oh, axis=0, keepdims=True)
            
            # 4. Update parameters
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        Z = np.dot(X, self.W) + self.b
        A = self.softmax(Z)
        return np.argmax(A, axis=1)

# ==========================================
# 3. Application / Execution
# ==========================================
np.random.seed(42)

# Generate Synthetic Data: 3 Distinct Clusters
# Class 0: centered at (2,2), Class 1: (6,6), Class 2: (2,8)
X_0 = np.random.randn(100, 2) + np.array([2, 2])
X_1 = np.random.randn(100, 2) + np.array([6, 6])
X_2 = np.random.randn(100, 2) + np.array([2, 8])

# Binary Dataset (Classes 0 and 1 only)
X_bin = np.vstack([X_0, X_1])
y_bin = np.hstack([np.zeros(100), np.ones(100)]).astype(int)

# Multiclass Dataset (Classes 0, 1, and 2)
X_multi = np.vstack([X_0, X_1, X_2])
y_multi = np.hstack([np.zeros(100), np.ones(100), np.full(100, 2)]).astype(int) 

# Standardizing the Binary Dataset
X_bin_mean = np.mean(X_bin, axis=0)
X_bin_std = np.std(X_bin, axis=0)
X_bin_scaled = (X_bin - X_bin_mean) / X_bin_std

# Standardizing the Multiclass Dataset
X_multi_mean = np.mean(X_multi, axis=0)
X_multi_std = np.std(X_multi, axis=0)
X_multi_scaled = (X_multi - X_multi_mean) / X_multi_std

# Then pass the scaled data to the fit functions:
# Train Binary Model
binary_model = BinaryLogisticRegression(learning_rate=0.1, epochs=500)
#binary_model.fit(X_bin, y_bin)
binary_model.fit(X_bin_scaled, y_bin)

# Train Multiclass Model
multi_model = MulticlassLogisticRegression(learning_rate=0.1, epochs=500)
#multi_model.fit(X_multi, y_multi)
multi_model.fit(X_multi_scaled, y_multi)

# ==========================================
# 4. Plotting the Gradient Descent Curves
# ==========================================
plt.figure(figsize=(14, 5))

# Plot 1: Binary LR Cost Curve
plt.subplot(1, 2, 1)
plt.plot(range(binary_model.epochs), binary_model.cost_history, color='blue', linewidth=2)
plt.title('Binary Logistic Regression\nGradient Descent Optimization')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Cost (Log Loss)')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Multiclass LR Cost Curve
plt.subplot(1, 2, 2)
plt.plot(range(multi_model.epochs), multi_model.cost_history, color='red', linewidth=2)
plt.title('Multiclass Logistic Regression (Softmax)\nGradient Descent Optimization')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Cost (Categorical Cross-Entropy)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "logistic_regression.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()