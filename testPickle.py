import numpy as numpy
import pickle

#Print the distribution of genres in the dataset

def load(id):
    with open('Data/TrainingData'+str(id)+'.pkl', 'rb') as f:
        return pickle.load(f)

res = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(1, 11):
    test = load(i)
    for _, val in test.items():
        index = val['genre'].tolist().index(1)
        res[index] = res[index] + 1

print(res)