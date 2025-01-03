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

#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from itertools import cycle

sns.set_theme(style='white', palette = None)
color_pal = plt.rcParams['axes.prop_cycle'].by_key()["color"]

audio_files = glob.glob('CREMA-D/AudioWAV/*.wav')
emotions = {"ANG": "anger",
            "DIS": "disgust",
            "FEA": "fear",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad"}

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

# x_train, x_test, y_train, y_test = LoadData()

# Lts do data normalization
# mean = np.mean(x_train, axis=0)
# std = np.std(x_train, axis=0)
#
# x_train = (x_train - mean)/std
# x_test = (x_test - mean)/std
# np.save("x_train",x_train)
# np.save("y_train",y_train)
# np.save("x_test",x_test)
# np.save("y_test",y_test)

def loadFromBinary(x_train, y_train, x_test, y_test):
    return np.load(x_train), np.load(x_test), np.load(y_train), np.load(y_test)

x_train, x_test, y_train, y_test = loadFromBinary("x_train.npy", "y_train.npy",
                                                  "x_test.npy", "y_test.npy")

print(x_train)

#Shape of train and test set and Number of features extracted
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')


model_basic = MLPClassifier(hidden_layer_sizes=(150,100,100,), activation="relu",
                            solver="adam", alpha=0.0001,
                            batch_size=300, learning_rate="constant",
                            learning_rate_init=0.001, max_iter=250,
                            shuffle=True, random_state=42,
                            tol=1e-4, verbose=True, early_stopping= False,
                            momentum=0.9, nesterovs_momentum=True,
                            beta_1=0.9, beta_2=0.999,
                            epsilon=1e-08, n_iter_no_change=50,
                            )


model_test = MLPClassifier(hidden_layer_sizes=(200,200,200,), activation="relu",
                            solver="adam", alpha=0.0001,
                            batch_size=100, learning_rate="constant",
                            learning_rate_init=0.001, max_iter=1000,
                            shuffle=True, random_state=42,
                            tol=1e-4, verbose=True,
                            momentum=0.9, nesterovs_momentum=True,
                            beta_1=0.9, beta_2=0.999,
                            epsilon=1e-08, n_iter_no_change=50,
                            )
model = model_test

#-------------------------------------------------------


scores_train = []
scores_test = []
n_epoch = 100
mini_batch_size = model.batch_size
# EPOCH
epoch = 0
while epoch < n_epoch:
    print('epoch: ', epoch)
    # SHUFFLING
    random_perm = np.random.permutation(x_train.shape[0])
    mini_batch_index = 0
    while True:
        # MINI-BATCH
        indices = random_perm[mini_batch_index:mini_batch_index + mini_batch_size]
        model.partial_fit(x_train[indices], y_train[indices], classes=np.unique(y_train))
        mini_batch_index += mini_batch_size

        if mini_batch_index >= x_train.shape[0]:
            break

    # SCORE TRAIN
    scores_train.append(model.score(x_train, y_train))

    # SCORE TEST
    scores_test.append(model.score(x_test, y_test))

    epoch += 1

""" Plot """

plt.plot(scores_train, color='green', alpha=0.8, label='Train')
plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
plt.title(f"Accuracy over epochs (Test (200,200,200) neurons, batch 100)", fontsize=14)
plt.xlabel('Epochs')
plt.legend(loc='upper left')
plt.show()


#--------------------------------------------------------

# model.fit(x_train, y_train)
#
# # Predict for the test set
# y_pred = model.predict(x_test)
#
# # Calculate Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: {:.2f}%".format(accuracy * 100))

plt.plot(model.loss_curve_)
if model.early_stopping:
    plt.plot(model.validation_scores_)
plt.title("Loss over mini epochs (Test (200,200,200) neurons, batch 100)", fontsize=14)
plt.xlabel('Mini Epochs')
plt.legend(loc='upper right')
plt.show()