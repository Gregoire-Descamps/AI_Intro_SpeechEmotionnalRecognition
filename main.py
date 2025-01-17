import os
import random
import numpy as np
import matplotlib.pylab as plt
import soundfile
import glob
import librosa

#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score




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
def extract_feature(file_name, mfcc, chroma, mel, augmentation = None):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if augmentation == "noise":
            noise = np.random.randn(len(X))
            X = X + 0.005 * noise
        elif augmentation == "pitch":
            pitch_factor = random.randint(-6,6)
            X = librosa.effects.pitch_shift(X, sr= sample_rate,n_steps=pitch_factor)
        elif augmentation == "stretch":
            stretch_factor = random.randint(8,12)/10
            X = librosa.effects.time_stretch(X,rate=stretch_factor)

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




def loadData(test_size =0.2, type = None):
    x, y = [], []
    for file in glob.glob('CREMA-D/AudioWAV/*.wav'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split('_')[2]]

        if emotion not in emotions.values():
            continue
        # match case to create a training dataset with  an addition of 20% of random augmented data
        if type == "mixed":
            x.append( extract_feature(file, mfcc=True, chroma=True, mel=True))
            y.append(emotion)

            if random.randint(0,4)<1 :
                match random.randint(1,3):
                    case 1:
                        x.append(extract_feature(file, mfcc=True, chroma=True, mel=True, augmentation="noise"))
                        y.append(emotion)
                    case 2:
                        x.append(extract_feature(file, mfcc=True, chroma=True, mel=True, augmentation="pitch"))
                        y.append(emotion)
                    case 3:
                        x.append(extract_feature(file, mfcc=True, chroma=True, mel=True, augmentation="stretch"))
                        y.append(emotion)
                    case _:
                        pass

        elif type == "noisy" :
            x.append(extract_feature(file, mfcc=True, chroma=True, mel=True, augmentation="noise"))
            y.append(emotion)

        elif type == "pitched" :
            x.append(extract_feature(file, mfcc=True, chroma=True, mel=True, augmentation="pitch"))
            y.append(emotion)

        elif type == "stretched" :
            x.append(extract_feature(file, mfcc=True, chroma=True, mel=True, augmentation="stretch"))
            y.append(emotion)

        elif type == None:
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)

        else:
            raise(f'Argument \"{type}\" for dataset type not recognized.')

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


def datasetGeneration():

    datasetTypes = [None, "noisy", "pitched", "stretched", "mixed"]

    for type in datasetTypes:
        x_train, x_test, y_train, y_test = loadData(type=type)

        # data normalization
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean)/std
        x_test = (x_test - mean)/std

        if type != None:
            datasetType = "_" + str(type)
        else:
            datasetType = ""

        np.save("x_train" + datasetType,x_train)
        np.save("y_train" + datasetType,y_train)
        np.save("x_test" + datasetType,x_test)
        np.save("y_test" + datasetType,y_test)


def loadFromBinary(x_train, y_train, x_test, y_test):
    return np.load(x_train), np.load(x_test), np.load(y_train), np.load(y_test)


# Fit model and plot accuracy on train and test data over 100 epochs
def PartialFit(model, n_epoch =100, minibatch_size = None, plot = True):
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

    if plot:
        plt.plot(scores_train, color='green', alpha=0.8, label='Train')
        plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
        plt.title(f"Accuracy over epochs (Test {model.solver}, {model.hidden_layer_sizes} , batch {model.batch_size},\n"
                  f"LR {model.learning_rate}({model.learning_rate_init})", fontsize=14)
        plt.xlabel('Epochs')
        plt.legend(loc='upper left')
        plt.show()
    return model



def predictScore(model):
    # Predict for the test set
    y_pred = model.predict(x_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return round(accuracy * 100, 2)

# Plot Loss and eventually validation score (only if early_stopping is enabled) of a model
def plotLoss(model):
    plt.plot(model.loss_curve_)
    if model.early_stopping:
        plt.plot(model.validation_scores_)
    plt.title(f"Loss over mini batch (Test {model.solver}, {model.hidden_layer_sizes} , batch {model.batch_size},\n"
              f"LR {model.learning_rate}({model.learning_rate_init})", fontsize=14)
    plt.xlabel('Mini Batch')
    plt.legend(loc='upper right')
    plt.show()

# Provide multiple model parameter lists to test and compare in order to find optimal parameters,
# return a dict of average accuracy results over parameter sets.
def optimalTuning(n_rounds = 10, solvers = ["adam"], hidden_layers_shapes = [(200,100,50)], batch_sizes = [300], learning_rates = ["constant"], learning_rate_inits = [0.0001], tols = [1e-4], n_iter_no_changes = [50]):
    results = {}
    for solver in solvers:
        results[solver]= {}
        for hidden_layers_shape in hidden_layers_shapes:
            results[solver][hidden_layers_shape]={}
            for batch_size in batch_sizes:
                results[solver][hidden_layers_shape][batch_size] = {}
                for learning_rate in learning_rates:
                    results[solver][hidden_layers_shape][batch_size][learning_rate] = {}
                    for learning_rate_init in learning_rate_inits:
                        results[solver][hidden_layers_shape][batch_size][learning_rate][learning_rate_init] = {}
                        for tol in tols:
                            results[solver][hidden_layers_shape][batch_size][learning_rate][learning_rate_init][tol] = {}
                            for n_iter_no_change in n_iter_no_changes:
                                accuracy_scores = []

                                model = MLPClassifier(hidden_layer_sizes=hidden_layers_shape, activation="relu",
                                        solver=solver, alpha=0.0001,
                                        batch_size=batch_size, learning_rate=learning_rate,
                                        learning_rate_init=learning_rate_init, max_iter=200,
                                        shuffle=True, random_state=None,
                                        tol=tol, verbose=False,
                                        momentum=0.9, nesterovs_momentum=True,
                                        beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, n_iter_no_change=n_iter_no_change,
                                        )
                                rnd = 0
                                while rnd < n_rounds:
                                    model.fit(x_train, y_train)
                                    accuracy_scores.append(predictScore(model))
                                    rnd += 1

                                results[solver][hidden_layers_shape][batch_size][learning_rate][learning_rate_init][tol][n_iter_no_change] = np.mean(accuracy_scores)
                                print(f"{solver}_{hidden_layers_shape}_{batch_size}_{learning_rate}_{learning_rate_init}_{tol}_{n_iter_no_changes} : {np.mean(accuracy_scores)}%")
    return results

# fit a model and diplay metrics
def directFit(model):
    fitted_model = model.fit(x_train, y_train)
    print(predictScore(fitted_model))
    plotLoss(fitted_model)



# datasetGeneration()


x_train, x_test, y_train, y_test = loadFromBinary("x_train_mixed.npy", "y_train_mixed.npy",
                                                  "x_test_mixed.npy", "y_test_mixed.npy")



#Shape of train and test set and Number of features extracted
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')


model_basic = MLPClassifier(hidden_layer_sizes=(200,100,50,), activation="relu",
                            solver="adam", alpha=0.0001,
                            batch_size=300, learning_rate="constant",
                            learning_rate_init=0.001, max_iter=250,
                            shuffle=True, random_state=42,
                            tol=1e-4, verbose=True, early_stopping= False,
                            momentum=0.9, nesterovs_momentum=True,
                            beta_1=0.9, beta_2=0.999,
                            epsilon=1e-08, n_iter_no_change=50,
                            )


model_test = MLPClassifier(hidden_layer_sizes=(200,200,200,200), activation="relu",
                            solver="adam", alpha=0.0001,
                            batch_size=300, learning_rate="constant",
                            learning_rate_init=0.001, max_iter=1000,
                            shuffle=True, random_state=42,
                            tol=1e-4, verbose=False,
                            momentum=0.9, nesterovs_momentum=True,
                            beta_1=0.9, beta_2=0.999,
                            epsilon=1e-08, n_iter_no_change=50,
                            )


#
model = model_test
# accuracies = []
# for i in range(3):
#     accuracies.append(optimalTuning(hidden_layers_shapes=[(200,100,50)], batch_sizes=[300]))
# print(accuracies)


# directFit(model)

# plot training loss on train dataset
plotLoss(PartialFit(model))