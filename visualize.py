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
from keras.models import Sequential, Model

import matplotlib.pyplot as plt

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

file = getMel(filename)
file = file.reshape(1, 64, 234, 1)

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

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(file)

result = model.predict(file).tolist()[0]

index = -1
entry = 0

for e in range(len(result)):
    if(result[e] > entry):
        entry = result[e]
        index = e

print("Estimated genre classification: ", genreLabels[index])

def displayActivation(activations, colSize, rowSize, actIndex): 
    activation = activations[actIndex]
    activationIndex=0
    fig, ax = plt.subplots(rowSize, colSize, figsize=(rowSize*2.5,colSize*1.5))
    for row in range(0,rowSize):
        for col in range(0,colSize):
            ax[row][col].imshow(activation[0, :, :, activationIndex], cmap='gray')
            activationIndex += 1

plt.imshow(file[0,:,:,0])

displayActivation(activations, 4, 2, 1)
displayActivation(activations, 4, 4, 2)
displayActivation(activations, 4, 4, 3)
displayActivation(activations, 4, 4, 4)
displayActivation(activations, 8, 4, 6)
displayActivation(activations, 8, 4, 9)

plt.show()