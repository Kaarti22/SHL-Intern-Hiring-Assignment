import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.extract_features import extract_mfcc_features

DATA_PATH = "data"
TEST_AUDIO_PATH = os.path.join(DATA_PATH, "audios_test")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
MODEL_PATH = "outputs/model_xgb.pkl"

model = joblib.load(MODEL_PATH)

test_df = pd.read_csv(TEST_CSV)

X_test = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    file_path = os.path.join(TEST_AUDIO_PATH, row["filename"])
    features = extract_mfcc_features(file_path)
    X_test.append(features)

X_test = np.array(X_test)

y_pred = model.predict(X_test)

submission_df = test_df.copy()
submission_df["label"] = y_pred

os.makedirs("outputs", exist_ok=True)
submission_df.to_csv("outputs/submission.csv", index=False)
print("Submission file saved to outputs/submission.csv")