import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ==========================================
# 1. Application dataset
# Tumor diagnosis: size (x1) vs texture (x2).
# Class 0 = benign, Class 1 = malignant. Train fits LDA, test is evaluated.
# ==========================================
np.random.seed(5)

def blob(center, n, spread):
    return np.random.randn(n, 2) * spread + np.array(center)

X0_train, X1_train = blob([2, 2], 40, 1.2), blob([5, 6], 40, 1.2)
X0_test, X1_test = blob([2, 2], 20, 1.2), blob([5, 6], 20, 1.2)
X_train = np.vstack([X0_train, X1_train])
y_train = np.hstack([np.zeros(40), np.ones(40)]).astype(int)
X_test = np.vstack([X0_test, X1_test])
y_test = np.hstack([np.zeros(20), np.ones(20)]).astype(int)

# ==========================================
# 2. LDA with sklearn (fit on TRAIN only)
# ==========================================
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

w = lda.scalings_[:, 0] / np.linalg.norm(lda.scalings_[:, 0])  # LDA direction
if (X1_train @ w).mean() < (X0_train @ w).mean():              # orient toward class 1
    w = -w

z_train = lda.transform(X_train).ravel() * np.sign(w @ lda.scalings_[:, 0])
z_test = lda.transform(X_test).ravel() * np.sign(w @ lda.scalings_[:, 0])
test_acc = float(lda.score(X_test, y_test) * 100)
print(f"LDA direction: [{w[0]:.3f}, {w[1]:.3f}]")
print(f"Test accuracy with 1 LDA feature: {test_acc:.1f}%")

threshold = round(float((z_train[y_train == 0].mean() + z_train[y_train == 1].mean()) / 2), 2) + 0.0
print(f"Threshold (midpoint of projected means): {threshold:.2f}")

anchor = (X0_train.mean(axis=0) + X1_train.mean(axis=0)) / 2
t_train = (X_train - anchor) @ w
t_test = (X_test - anchor) @ w
X_train_proj = anchor + np.outer(t_train, w)
X_test_proj = anchor + np.outer(t_test, w)

# ==========================================
# 3. Plots: projected line + reduced features on it
# ==========================================
plt.figure(figsize=(9, 5))

# 2D data, LDA line, every point -> its mapped point on the line
plt.scatter(X0_train[:, 0], X0_train[:, 1], c="steelblue", label="Train benign (0)", alpha=0.7)
plt.scatter(X1_train[:, 0], X1_train[:, 1], c="darkorange", label="Train malignant (1)", alpha=0.7)
plt.scatter(X0_test[:, 0], X0_test[:, 1], c="steelblue", marker="^", s=60,
            edgecolors="k", linewidths=0.5, label="Test benign (0)")
plt.scatter(X1_test[:, 0], X1_test[:, 1], c="darkorange", marker="^", s=60,
            edgecolors="k", linewidths=0.5, label="Test malignant (1)")
s = np.linspace(t_train.min() - 0.5, t_train.max() + 0.5, 100)
plt.plot(anchor[0] + s * w[0], anchor[1] + s * w[1], "r-", linewidth=2,
         label="LDA projected line")
for p, q in zip(X_train, X_train_proj):
    plt.plot([p[0], q[0]], [p[1], q[1]], "gray", linewidth=0.7, alpha=0.4)
for p, q in zip(X_test, X_test_proj):
    plt.plot([p[0], q[0]], [p[1], q[1]], "k-", linewidth=1.0, alpha=0.7)
plt.scatter(X_train_proj[:, 0], X_train_proj[:, 1], c="steelblue", s=20,
            marker="s", label="Train mapped on LDA", zorder=4)
plt.scatter(X_test_proj[:, 0], X_test_proj[:, 1], c="red", s=30,
            label="Test mapped on LDA", zorder=4)
plt.title(f"LDA: Points Projected onto Discriminant Line (test acc={test_acc:.1f}%)")
plt.xlabel("Tumor size (x1)")
plt.ylabel("Texture (x2)")
plt.legend(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lda.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()
