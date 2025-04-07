import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from features.generate_mel_spectrograms import save_mel_spectrogram

DATA_PATH = "data"
AUDIO_FOLDER = os.path.join(DATA_PATH, "audios_train")
CSV_FILE = os.path.join(DATA_PATH, "train.csv")
OUTPUT_FOLDER = os.path.join(DATA_PATH, "mel_spectrograms_train")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_FILE)

for _, row in df.iterrows():
    filename = row["filename"]
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".wav", ".png"))
    save_mel_spectrogram(audio_path, output_path)
