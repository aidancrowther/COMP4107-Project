import os
import tensorflow as tf
import numpy as np
import scipy as sp
import pickle
import matplotlib.pyplot as plt
import random
from time import time

import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ConvLSTM2D, LSTM, Reshape, Dropout, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

#Setup keras callbacks for checkpointing, earlystopping, and tensorboard
filepath="Data/weights-best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=6, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
callbacks_list = [earlystop, tensorboard, checkpoint]

#Initialize dimensions and datasets
dim1 = 0
dim2 = 0
trainingData, trainingLabels, testingData, testingLabels = None, None, None, None

#Load a pickle file given its encoded ID
def load(batchId):
    with open('Data/TrainingData'+str(batchId)+'.pkl', 'rb') as f:
        return pickle.load(f)

#return batches and labels from the encoded data
def getBatch(batchId):

    batch = load(batchId)
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

#Load all training/testing data from the 'Data' directory
def prepData():

    global trainingData
    global trainingLabels
    global testingData
    global testingLabels
    global dim1
    global dim2

    trainingData, trainingLabels, dim1, dim2 = getBatch(1)
    testingData, testingLabels, _, _ = getBatch(7)

    for i in range(2, 7):
        tempTrain, tempLabels, _, _ = getBatch(i)
        trainingData = np.concatenate((trainingData, tempTrain))
        trainingLabels = np.concatenate((trainingLabels, tempLabels))

    testingData, testingLabels, _, _ = getBatch(10)

    return

prepData()

#Setup the network, we have 8 classifiers
num_classes = 8

#Use a sequential model
model = Sequential()

#Add a convolutional layer with 8 filters of kernel size (6x6)
model.add(Conv2D(8, kernel_size=(6, 6), strides=(1, 1), input_shape=(dim1, dim2, 1)))
#Pool using a (4x4) mask
model.add(MaxPooling2D(pool_size=(4, 4)))
#Convolutional layer with 16 filters of (4x4)
model.add(Conv2D(16, (4, 4)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
#Pool using a (2x2) mask
model.add(MaxPooling2D(pool_size=(2, 2)))
#Convolutional layer with 32 filters of (3x3)
model.add(Conv2D(32, (3, 3)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
#Convolutional layer with 32 filters of (2x2)
model.add(Conv2D(32, (2, 2)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
#Reshape the layer to be converted for the RNN
model.add(Reshape(target_shape=(48, 32)))
#Use a ConvLSTM2D to connect the CNN and RNN
ConvLSTM2D(filters=8, kernel_size=(3, 3), input_shape=(None, 110, 64), padding='same', return_sequences=True,  stateful = True)
#First LSTM layer with 64 elements
model.add(LSTM(64, return_sequences=True, input_shape=(110, 64)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
#LSTM layer with 32 elements
model.add(LSTM(32, activation='relu', return_sequences=True))
#Flatten the network to be used with the dense layer
model.add(Flatten())
#Dense layer with 1024 neurons
model.add(Dense(1024))
model.add(Dropout(0.3))
model.add(Activation("relu"))
#Dense layer with 16 neurons
model.add(Dense(16, activation='relu'))
#Output layer with 8 classifications
model.add(Dense(num_classes, activation='softmax'))

#Print the model summary
print(model.summary())

#Compile the keras network using the adam optimizer, categorical cross entropy, and tracking network accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Run the network with the training and testing data, run for up to 100 epochs, feeding in the callbacks we defined
model.fit(trainingData, trainingLabels,
          batch_size=128,
          epochs=100,
          verbose=1,
          validation_data=(testingData, testingLabels),
          shuffle = True,
          callbacks=callbacks_list)