import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Application dataset (from scratch)
# Concentric circles: inner blob = class 0, outer ring = class 1.
# Small training set + high-degree polynomial features => overfits
# without regularization.
# ==========================================
np.random.seed(11)

def make_circles(n_inner=25, n_ring=25, noise=0.6, flip=0.05):
    r_inner = np.abs(np.random.randn(n_inner)) * 0.9
    t_inner = np.random.rand(n_inner) * 2 * np.pi
    X0 = np.c_[r_inner * np.cos(t_inner), r_inner * np.sin(t_inner)]
    r_ring = 2.5 + np.random.randn(n_ring) * noise
    t_ring = np.random.rand(n_ring) * 2 * np.pi
    X1 = np.c_[r_ring * np.cos(t_ring), r_ring * np.sin(t_ring)]
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_inner), np.ones(n_ring)]).astype(int)
    noisy = np.random.rand(len(y)) < flip  # label noise: forces memorization
    y[noisy] = 1 - y[noisy]
    return X, y

X_train, y_train = make_circles()
X_test, y_test = make_circles(n_inner=60, n_ring=60)

# Polynomial feature expansion (degree 8, no inbuilt function)
DEGREE = 8

def poly_features(X):
    x1, x2 = X[:, 0], X[:, 1]
    cols = [np.ones(len(X))]
    for d in range(1, DEGREE + 1):
        for k in range(d + 1):
            cols.append((x1 ** (d - k)) * (x2 ** k))
    return np.column_stack(cols)

Phi_train = poly_features(X_train)
Phi_test = poly_features(X_test)

# Standardize using training statistics (skip bias column)
mu = Phi_train[:, 1:].mean(axis=0)
sd = Phi_train[:, 1:].std(axis=0) + 1e-12
Phi_train[:, 1:] = (Phi_train[:, 1:] - mu) / sd
Phi_test[:, 1:] = (Phi_test[:, 1:] - mu) / sd

# ==========================================
# 2. Logistic Regression with L2 (from scratch)
# Cost = BCE + (lambd/2m) * ||W||^2 ; gradient gets + (lambd/m)*W
# ==========================================
class LogisticL2:
    def __init__(self, learning_rate=0.5, epochs=3000, lambd=0.0):
        self.lr = learning_rate
        self.epochs = epochs
        self.lambd = lambd
        self.cost_history = []

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        y = y.reshape(-1, 1)
        self.W = np.zeros((n, 1))
        self.b = 0.0
        for _ in range(self.epochs):
            A = self.sigmoid(X @ self.W + self.b)
            eps = 1e-15
            bce = -(1 / m) * np.sum(y * np.log(A + eps) + (1 - y) * np.log(1 - A + eps))
            reg = (self.lambd / (2 * m)) * np.sum(self.W[1:] ** 2)  # don't penalize bias
            self.cost_history.append(bce + reg)
            dW = (1 / m) * (X.T @ (A - y))
            dW[1:] += (self.lambd / m) * self.W[1:]
            db = (1 / m) * np.sum(A - y)
            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict(self, X):
        return (self.sigmoid(X @ self.W + self.b) >= 0.5).astype(int).ravel()

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y) * 100)

# ==========================================
# 3. Train before (lambd=0) vs after (lambd>0) regularization
# ==========================================
model_plain = LogisticL2(lambd=0.0).fit(Phi_train, y_train)
model_reg = LogisticL2(lambd=0.5).fit(Phi_train, y_train)

acc = {
    "plain_train": model_plain.accuracy(Phi_train, y_train),
    "plain_test": model_plain.accuracy(Phi_test, y_test),
    "reg_train": model_reg.accuracy(Phi_train, y_train),
    "reg_test": model_reg.accuracy(Phi_test, y_test),
}
w_plain = float(np.abs(model_plain.W[1:]).sum())
w_reg = float(np.abs(model_reg.W[1:]).sum())
print(f"Without regularization : train={acc['plain_train']:.1f}%  test={acc['plain_test']:.1f}%  |W|={w_plain:.0f}")
print(f"With L2 (lambd=0.5)    : train={acc['reg_train']:.1f}%  test={acc['reg_test']:.1f}%  |W|={w_reg:.0f}")
print(f"Overfitting gap        : {acc['plain_train']-acc['plain_test']:.1f}% -> "
      f"{acc['reg_train']-acc['reg_test']:.1f}%")

# ==========================================
# 4. Plots: decision boundaries + performance before/after
# ==========================================
def boundary_grid(model, h=0.05):
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Phi = poly_features(np.c_[xx.ravel(), yy.ravel()])
    Phi[:, 1:] = (Phi[:, 1:] - mu) / sd
    return xx, yy, model.predict(Phi).reshape(xx.shape)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
xx, yy, Z = boundary_grid(model_plain)
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k")
plt.title(f"Decision Boundary WITHOUT Regularization\ntrain={acc['plain_train']:.1f}% test={acc['plain_test']:.1f}%")
plt.xlabel("x1")
plt.ylabel("x2")

plt.subplot(1, 3, 2)
xx, yy, Z = boundary_grid(model_reg)
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k")
plt.title(f"Decision Boundary WITH L2 Regularization\ntrain={acc['reg_train']:.1f}% test={acc['reg_test']:.1f}%")
plt.xlabel("x1")
plt.ylabel("x2")

plt.subplot(1, 3, 3)
labels = ["Without\nRegularization", "With L2\nRegularization"]
x = np.arange(len(labels))
w = 0.35
plt.bar(x - w / 2, [acc["plain_train"], acc["reg_train"]], w, label="Train accuracy")
plt.bar(x + w / 2, [acc["plain_test"], acc["reg_test"]], w, label="Test accuracy")
plt.xticks(x, labels)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.title("Performance Before vs After Regularization")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "regularization.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()
