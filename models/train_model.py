import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.extract_features import extract_all_features

DATA_PATH = "data"
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
audio_path = os.path.join(DATA_PATH, "audios_train")

X, y = [], []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting features"):
    filepath = os.path.join(audio_path, row["filename"])
    features = extract_all_features(filepath)
    X.append(features)
    y.append(row["label"])

X, y = np.array(X), np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

os.makedirs("outputs", exist_ok=True)
joblib.dump(scaler, "outputs/scaler.pkl")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

print("Starting GridSearcgCV...")
grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_squared_error", verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Parameters: ", grid.best_params_)

y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
pearson_corr, _ = pearsonr(y_val, y_pred)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation Pearson Correlation: {pearson_corr:.4f}")

os.makedirs("outputs", exist_ok=True)
joblib.dump(best_model, "outputs/model_xgb.pkl")
print("Model saved to outputs/model_xgb.pkl")

sns.set(style="whitegrid")

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_val, y=y_pred)
plt.xlabel("Actual Grammar Score")
plt.ylabel("Predicted Grammar Score")
plt.title("Actual vs Predicted Scores")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

residuals = y_val - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, bins=20, color='purple')
plt.title("Residuals Distribution (Actual - Predicted)")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/residual_distribution.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(y_pred, kde=True, bins=20, color='teal')
plt.title("Distribution of Predicted Grammar Scores")
plt.xlabel("Predicted Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/predicted_score_distribution.png")
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_pred, alpha=0.7)
z = np.polyfit(y_val, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_val, p(y_val), color="red", linestyle="--", label="Best fit line")
plt.xlabel("Actual Grammar Score")
plt.ylabel("Predicted Grammar Score")
plt.title("Actual vs Predicted with Best Fit Line")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted_with_line.png")
plt.show()