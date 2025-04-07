import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.extract_features import extract_mfcc_features

DATA_PATH = "data"
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
audio_path = os.path.join(DATA_PATH, "audios_train")

X, y = [], []
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    filepath = os.path.join(audio_path, row["filename"])
    features = extract_mfcc_features(filepath)
    X.append(features)
    y.append(row["label"])

X, y = np.array(X), np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
pearson_corr, _ = pearsonr(y_val, y_pred)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation Pearson Correlation: {pearson_corr:.4f}")

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model_xgb.pkl")
print("Model saved to outputs/model_xgb.pkl")