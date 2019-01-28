import os
import csv
import ast
import numpy as np
import librosa
import imageio
import pickle
import warnings
import time

#Convert all mp3's in the Data/Songs/ directory into MEL spectrograms and encode them to pickle files for distribution

#Converted files use 16000Hz sample rate and 64 mels

warnings.filterwarnings("ignore")

dimToFind = 0
found = 0

songs = []
metadata = []
genres = [2, 10, 12, 15, 17, 21, 38, 1235]
genreLabels = ["International", "Pop", "Rock", "Electronic", "Folk", "Hip-Hop", "Experimental", "Instrumental"]

def millis():
    return int(round(time.time() * 1000))

def pad(array, reference_shape, offsets):
    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def save(obj, val):
    with open('Data/TrainingData'+str(val)+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def oneHotEncode(x):
    encoded = np.zeros((len(x), 8))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def getMel(filename, genre):

    global found
    global dimToFind

    while(len(filename) < 6):
        filename = '0'+filename

    audio_path = 'Data/Songs/'+filename+'.mp3'

    sr = 16000
    numMels = 64

    y, _ = librosa.load(audio_path, sr=sr)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=numMels)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = normalize(log_S)

    if(dimToFind == 0):
        dimToFind = log_S.shape
        dimToFind = (dimToFind[0], dimToFind[1]+3)

    log_S = pad(log_S, dimToFind, (0, 0))

    if(log_S.shape == dimToFind): found += 1

    imageio.imwrite('Data/Datagrams/'+filename+'.png', log_S)
    return filename, {'data' : log_S, 'genre' : genre}

def loadData():

    for _, _, files in os.walk("./Data/Songs"):  
        for filename in files:
            songs.append(filename.split(".")[0].lstrip('0'))

    songs.sort(key=int)

    with open('Data/Metadata/tracks.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if(row[0] in songs):
                metadata.append(genreLabels.index(row[40]))

    encodedMetadata = oneHotEncode(metadata)

    print(len(songs))
    print(encodedMetadata.shape)

    percentage = 0
    dataCount = 0
    finalDict = {}
    start = millis()
    curr = millis()

    for i in range(len(songs)):
        if(i%80 == 0):
            curr = millis()
            timeLeft = ((curr-start)/1000)*(100-percentage)
            print(percentage,"%, ETA: ",timeLeft," seconds remaining")
            percentage += 1
            start = millis()

        if(i%800 == 0 and i > 0):
            dataCount += 1
            save(finalDict, dataCount)
            finalDict = {}
            print("Saved")

        entryKey, entryVal = getMel(songs[i], encodedMetadata[i])
        finalDict.update({entryKey : entryVal})

    dataCount += 1
    save(finalDict, dataCount)
    finalDict = {}
    print("Saved")
    print("Found ",found, dimToFind, " dimension results")

loadData()