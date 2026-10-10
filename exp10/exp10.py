import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. Application dataset
# Study hours (x1) vs exam score (x2), 30 points.
# ==========================================
np.random.seed(11)

hours = np.random.uniform(1, 10, 30)
scores = 3 * hours + 20 + np.random.normal(0, 9, 30)
X = np.c_[hours, scores]

# Standardize (so both features count equally)
mu = X.mean(axis=0)
sd = X.std(axis=0)
Xs = (X - mu) / sd

# ==========================================
# 2. PCA with sklearn
# ==========================================
pca = PCA(n_components=1)
z = pca.fit_transform(Xs).ravel()              # new 1D reduced feature
pc1 = pca.components_[0]
var_ratio = pca.explained_variance_ratio_
X_proj = pca.inverse_transform(z.reshape(-1, 1)) * sd + mu  # mapped on PC1 line

print(f"PC1 direction: [{pc1[0]:.3f}, {pc1[1]:.3f}]")
print(f"Explained variance: PC1={var_ratio[0]*100:.1f}%")
mse = float(np.mean((X - X_proj) ** 2))
print(f"Reconstruction MSE (1 component): {mse:.2f}")

# ==========================================
# 3. Plots: projection onto PC1 + explained variance
# ==========================================
plt.figure(figsize=(14, 5))

# Plot 1: original points -> lines -> new points on the PC1 line
# (drawn in STANDARDIZED units, where PCA operates, so drops are orthogonal)
plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 1], c="steelblue", label="Standardized data", zorder=3)
s = np.linspace(z.min() - 0.5, z.max() + 0.5, 100)
plt.plot(s * pc1[0], s * pc1[1], "r-", linewidth=2, label="PC1 (projected line)")
plt.scatter([0], [0], c="black", s=80, marker="x", label="Mean", zorder=5)
Xs_proj = np.outer(z, pc1)
for p, q in zip(Xs, Xs_proj):  # perpendicular projection: point -> new point on PC1
    plt.plot([p[0], q[0]], [p[1], q[1]], "k-", linewidth=1.0, alpha=0.6)
plt.scatter(Xs_proj[:, 0], Xs_proj[:, 1], c="red", s=35,
            label="Reduced feature mapped on PC1", zorder=4)
plt.title("PCA: Points Projected Orthogonally onto PC1")
plt.xlabel("Standardized study hours (xs1)")
plt.ylabel("Standardized exam score (xs2)")
plt.legend(fontsize=9)
plt.grid(True, linestyle="--", alpha=0.6)
plt.axis("equal")

# Plot 2: explained variance
plt.subplot(1, 2, 2)
plt.bar(["PC1", "PC2"], [var_ratio[0] * 100, 100 - var_ratio[0] * 100],
        color=["red", "gray"])
plt.ylabel("Explained variance (%)")
plt.ylim(0, 105)
plt.title(f"Explained Variance (PC1={var_ratio[0]*100:.1f}%)")
plt.grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()
