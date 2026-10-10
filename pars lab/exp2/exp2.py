from flask import Flask, render_template
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)

# Load dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target
feature_names = housing.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest for feature extraction
feature_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

feature_model.fit(X_train, y_train)

# Feature importance
importance = feature_model.feature_importances_

feature_importance = sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
)

# Select relevant features
selected_features = [
    feature for feature, score in feature_importance
    if score > 0.05
]

# Train final model using selected features
selected_indices = [
    feature_names.index(feature)
    for feature in selected_features
]

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_selected, y_train)

# Predictions
predictions = model.predict(X_test_selected)

# Performance metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)


@app.route("/")
def index():

    # Data for feature importance graph
    feature_labels = [x[0] for x in feature_importance]
    feature_values = [round(x[1], 4) for x in feature_importance]

    # Performance data
    performance = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    }

    return render_template(
        "index.html",
        total_features=len(feature_names),
        selected_features=selected_features,
        feature_labels=feature_labels,
        feature_values=feature_values,
        performance=performance
    )


if __name__ == "__main__":
    app.run(debug=True)