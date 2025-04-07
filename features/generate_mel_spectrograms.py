import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

def save_mel_spectrogram(filepath, out_path, sr=16000, n_mels=128):
    y, sr = librosa.load(filepath, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout(pad=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()