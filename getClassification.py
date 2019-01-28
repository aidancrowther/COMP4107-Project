import sys
import librosa
import numpy as np
import imageio
import random
import pickle

import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ConvLSTM2D, LSTM, Reshape, Dropout, Activation
from keras.models import Sequential

#Return a classification for the input audio file using the trained network

genreLabels = ["International", "Pop", "Rock", "Electronic", "Folk", "Hip-Hop", "Experimental", "Instrumental"]

def trimToSize(song):
    selected = random.randint(0, len(song[0, :])-234)
    return song[:, selected : (selected+234)]

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

#Load a pickle file given its encoded ID
def load(fileToLoad):
    with open(fileToLoad, 'rb') as f:
        return pickle.load(f)

#return batches and labels from the encoded data
def getBatch(fileToLoad):

    batch = load(fileToLoad)
    batchOut = []
    labelsOut = []

    #Shuffle the keys
    keys =  list(batch.keys())
    random.shuffle(keys)

    #Augment the files by splitting them in 4, and adding all the data to the batch and label lists
    for key in keys:
        for i in range(4):
            batchOut.append(batch[key]['data'][:, (i*235) : (((i+1)*235)-1)])
            labelsOut.append(batch[key]['genre'])

    #Convert the arrays to np arrays
    batchOut, labelsOut = np.array(batchOut), np.array(labelsOut)

    #Reshape the data for the input tensor
    dim1, dim2 = batchOut[0].shape
    batchOut = batchOut.reshape(batchOut.shape[0], dim1, dim2, 1)

    #Return all batch, label, and dimension data
    return batchOut, labelsOut, dim1, dim2

def getMel(filename):

    while(len(filename) < 6):
        filename = '0'+filename

    audio_path = filename

    sr = 16000
    numMels = 64

    y, _ = librosa.load(audio_path, sr=sr)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=numMels)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = normalize(log_S)

    log_S = trimToSize(log_S)

    imageio.imwrite(filename+'.png', log_S)

    return log_S

filename = sys.argv[1]

validating = False
file = None
labels = None

if(filename.split(".")[1] == "mp3"):
    file = getMel(filename)
    file = file.reshape(1, 64, 234, 1)
elif(filename.split(".")[1] == "pkl"):
    file, labels, _, _ = getBatch(filename)
    file = file.reshape(file.shape[0], 64, 234, 1)
    validating = True
else:
    exit()

num_classes = 8

model = Sequential()
model.add(Conv2D(8, kernel_size=(6, 6), strides=(1, 1),
                 input_shape=(64, 234, 1)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, (4, 4)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(Conv2D(32, (2, 2)))
model.add(Dropout(0.3))
model.add(Activation("relu"))

model.add(Reshape(target_shape=(48, 32)))
ConvLSTM2D(filters=8, kernel_size=(3, 3), input_shape=(None, 110, 64), padding='same', return_sequences=True,  stateful = True)
model.add(LSTM(64, return_sequences=True, input_shape=(110, 64)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(Dense(16, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

model.load_weights("Data/weights-best.hdf5")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if(not validating):
    result = model.predict(file).tolist()[0]

    index = -1
    entry = 0

    for e in range(len(result)):
        if(result[e] > entry):
            entry = result[e]
            index = e

    print("Estimated genre classification: ", genreLabels[index])

else:
    result = model.predict(file).tolist()
    labelsToUse = labels.tolist()
    accuracy = 0.0
    count = 0

    for e in range(len(result)):
        if(np.argmax(result[e]) == labelsToUse[e].index(1.0)):
            accuracy += 1
        count += 1

    print("Validation accuracy: ", ((accuracy/count)*100),"%")