import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import soundfile

import glob

import librosa
import librosa.display
import IPython.display as ipd

from sklearn.model_selection import train_test_split

from itertools import cycle

sns.set_theme(style='white', palette = None)
color_pal = plt.rcParams['axes.prop_cycle'].by_key()["color"]

audio_files = glob.glob('CREMA-D/AudioWAV/*.wav')
emotions = {"ANG" :"anger",
            "DIS":"disgust",
            "FEA":"fear",
            "NEU":"neutral",
            "HAP":"happy",
            "SAD":"sad"}

emotions_level = {"XX":0,
                  "LO": 1,
                  "MD":2,
                  "HI": 3}


# Function for extracting mcc, chroma, and mel features from sound File
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel =np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result=np.hstack((result, mel))
    return result




def LoadData(test_size =0.2):
    x, y = [], []
    for file in glob.glob('CREMA-D/AudioWAV/*.wav'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split('_')[2]]

        if emotion not in emotions.values():
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size= test_size, random_state = 9)

print(LoadData())