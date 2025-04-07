import librosa
import numpy as np
import os

def extract_mfcc_features(filepath, n_mfcc=20):
    y, sr = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = {
        'mean': np.mean(mfcc, axis=1),
        'std': np.std(mfcc, axis=1),
        'min': np.min(mfcc, axis=1),
        'max': np.max(mfcc, axis=1)
    }
    return np.concatenate([features['mean'], features['std'], features['min'], features['max']])