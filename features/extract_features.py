import librosa
import numpy as np
import os

def extract_all_features(filepath, sr=16000):
    y, _ = librosa.load(filepath, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    features = np.concatenate([mfcc_mean, chroma_mean, contrast_mean, tonnetz_mean])
    return features