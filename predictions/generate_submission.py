import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.extract_features import extract_all_features

DATA_PATH = "data"
TEST_CSV_PATH = os.path.join(DATA_PATH, "test.csv")
AUDIO_PATH = os.path.join(DATA_PATH, "audios_test")
SCALER_PATH = "outputs/scaler.pkl"
MODEL_PATH = "outputs/model_xgb.pkl"
SUBMISSION_PATH = "outputs/submission.csv"

test_df = pd.read_csv(TEST_CSV_PATH)

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

X_test = []
filenames = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting test features"):
    filepath = os.path.join(AUDIO_PATH, row["filename"])
    features = extract_all_features(filepath)
    X_test.append(features)
    filenames.append(row["filename"])

X_test = np.array(X_test)
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

submission_df = pd.DataFrame({
    "filename": filenames,
    "label": y_pred
})

os.makedirs("outputs", exist_ok=True)
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file saved to {SUBMISSION_PATH}")