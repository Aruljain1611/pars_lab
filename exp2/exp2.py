from flask import Flask, request, render_template_string
import io
import os
import base64
import numpy as np
import matplotlib
# Use 'Agg' backend so matplotlib doesn't try to open GUI windows in Flask
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "datasets")
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# Datasets studied in Experiment 1
# ==========================================
DATASETS = {
    "breast_cancer": {
        "title": "Breast Cancer Wisconsin (Diagnostic)",
        "domain": "Healthcare / Medical Diagnosis",
        "loader": load_breast_cancer,
        "purpose": "Classify tumours as malignant or benign from nuclear features.",
    },
    "wine": {
        "title": "Wine Recognition",
        "domain": "Food Chemistry / Chemometrics",
        "loader": load_wine,
        "purpose": "Identify the cultivar of origin from chemical analysis.",
    },
}

# ==========================================
# Helper: Convert Matplotlib Figure to Base64
# ==========================================
def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode('utf-8')


def load_and_store(key):
    """Extract the dataset and store it as CSV (extract + store)."""
    info = DATASETS[key]
    data = info["loader"]()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target)
    csv_path = os.path.join(DATA_DIR, f"{key}.csv")
    if not os.path.exists(csv_path):
        header = ",".join(list(data.feature_names) + ["target"])
        np.savetxt(csv_path, np.c_[X, y], delimiter=",", header=header, comments="")
    return data, X, y, csv_path


def multiclass_roc(ax, y_test, y_prob, target_names):
    """One-vs-rest ROC curves (works for binary and 3-class)."""
    n_classes = y_prob.shape[1]
    colors = ["crimson", "darkgreen", "darkorange"]
    for i in range(n_classes):
        binary_true = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f'{target_names[i]} (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

# ==========================================
# HTML Template (with inline CSS)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Feature Extraction & Performance</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8e9f0; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #4a235a; }
        .tabs { text-align: center; margin-bottom: 20px; }
        .tabs a { display: inline-block; padding: 10px 22px; margin: 0 6px; border-radius: 8px; text-decoration: none; font-weight: bold; color: #4a235a; background: white; border: 2px solid #8e44ad; }
        .tabs a.active { background: #8e44ad; color: white; }
        .stats-panel { display: flex; justify-content: space-between; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 4px solid #8e44ad; margin-bottom: 30px; }
        .stat-box { text-align: center; flex: 1; }
        .stat-box:not(:last-child) { border-right: 1px solid #eee; }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #8e44ad; }
        .stat-label { color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; margin-top: 5px; }

        .charts-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between; }
        .chart-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); flex: 1; min-width: 350px; text-align: center; }
        .chart-card.full-width { flex: 100%; }
        .chart-card img { max-width: 100%; height: auto; border-radius: 4px; }
        .chart-title { font-weight: bold; font-size: 1.2em; color: #34495e; margin-bottom: 15px; }

        table.feat { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.85em; background: white; }
        table.feat th, table.feat td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        table.feat th { background: #8e44ad; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Feature Extraction & Performance Analysis</h1>
            <p>Dataset: {{ info.title }} | Domain: {{ info.domain }} | {{ info.purpose }}</p>
        </div>

        <div class="tabs">
            {% for key, ds in datasets.items() %}
            <a href="/?dataset={{ key }}" class="{{ 'active' if key == active else '' }}">{{ ds.title }}</a>
            {% endfor %}
        </div>

        <div class="stats-panel">
            <div class="stat-box">
                <div class="stat-value">{{ results.original_features }}</div>
                <div class="stat-label">Original Features</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ results.extracted_features }}</div>
                <div class="stat-label">Extracted Features (PCA)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ results.variance_explained }}</div>
                <div class="stat-label">Data Variance Retained</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #27ae60;">{{ results.accuracy }}</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
        </div>

        <div class="charts-container">
            <div class="chart-card full-width">
                <div class="chart-title">1. Feature Extraction: Dimensionality Reduction</div>
                <img src="data:image/png;base64,{{ results.plot_pca }}" alt="PCA Scatter Plot">
            </div>
            <div class="chart-card">
                <div class="chart-title">2. Model Performance: Confusion Matrix</div>
                <img src="data:image/png;base64,{{ results.plot_cm }}" alt="Confusion Matrix">
            </div>
            <div class="chart-card">
                <div class="chart-title">3. Model Performance: ROC Curve</div>
                <img src="data:image/png;base64,{{ results.plot_roc }}" alt="ROC Curve">
            </div>
        </div>

        <div class="chart-card full-width" style="margin-top:20px;">
            <div class="chart-title">4. Feature Description (mean / std / min / max on stored data)</div>
            <table class="feat">
                <tr><th>#</th><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                {% for f in features %}
                <tr><td>{{ loop.index }}</td><td>{{ f.name }}</td><td>{{ f.mean }}</td>
                    <td>{{ f.std }}</td><td>{{ f.min }}</td><td>{{ f.max }}</td></tr>
                {% endfor %}
            </table>
            <p style="color:#7f8c8d;">Target classes: {{ target_names | join(', ') }}</p>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# Flask Route & ML Pipeline
# ==========================================
@app.route('/')
def index():
    key = request.args.get('dataset', 'breast_cancer')
    if key not in DATASETS:
        key = 'breast_cancer'
    info = DATASETS[key]

    # 1. Extract + store the dataset
    data, X, y, csv_path = load_and_store(key)
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    n_samples, n_features = X.shape

    # 2. Standardization
    X_scaled = StandardScaler().fit_transform(X)

    # 3. Feature Extraction (PCA -> 2 components)
    pca = PCA(n_components=2)
    X_extracted = pca.fit_transform(X_scaled)
    variance_explained = sum(pca.explained_variance_ratio_) * 100

    # 4. Train/Test Split & Model Training
    X_train, X_test, y_train, y_test = train_test_split(
        X_extracted, y, test_size=0.3, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 5. Performance Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Plot 1: PCA Scatter
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    scatter = ax1.scatter(X_extracted[:, 0], X_extracted[:, 1], c=y,
                          cmap='viridis', edgecolors='k', alpha=0.8)
    ax1.set_title("Feature Extraction: PCA (2 Components)")
    ax1.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    handles, _ = scatter.legend_elements(prop="colors")
    ax1.legend(handles, target_names, title="Class")
    ax1.grid(True, linestyle='--', alpha=0.6)
    plot_pca = fig_to_base64(fig1)

    # Plot 2: Confusion Matrix
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    im = ax2.imshow(cm, cmap='Purples')
    fig2.colorbar(im, ax=ax2)
    ax2.set_xticks(range(len(target_names)), target_names)
    ax2.set_yticks(range(len(target_names)), target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                     fontsize=14, fontweight="bold")
    ax2.set_title("Performance: Confusion Matrix")
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("True Class")
    plot_cm = fig_to_base64(fig2)

    # Plot 3: ROC Curve (one-vs-rest, works for 2 or 3 classes)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    multiclass_roc(ax3, y_test, y_prob, target_names)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Receiver Operating Characteristic (ROC)')
    ax3.legend(loc="lower right")
    ax3.grid(True, linestyle='--', alpha=0.6)
    plot_roc = fig_to_base64(fig3)

    features = [{
        "name": name,
        "mean": f"{X[:, j].mean():.2f}",
        "std": f"{X[:, j].std():.2f}",
        "min": f"{X[:, j].min():.2f}",
        "max": f"{X[:, j].max():.2f}",
    } for j, name in enumerate(feature_names)]

    results = {
        'original_features': n_features,
        'extracted_features': X_extracted.shape[1],
        'variance_explained': f"{variance_explained:.2f}%",
        'accuracy': f"{accuracy:.2f}%",
        'plot_pca': plot_pca,
        'plot_cm': plot_cm,
        'plot_roc': plot_roc,
    }

    return render_template_string(
        HTML_TEMPLATE, info=info, datasets=DATASETS, active=key,
        csv_path=csv_path, n_samples=n_samples, n_features=n_features,
        n_classes=len(target_names), results=results,
        features=features, target_names=target_names)


if __name__ == '__main__':
    print("Starting Data Visualization App (Exp1 datasets)...")
    print("Access the app at: http://127.0.0.1:5000")
    app.run(debug=True)
