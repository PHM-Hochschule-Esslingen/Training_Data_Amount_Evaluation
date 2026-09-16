from Nets_Only_Pytorch import MLP
from functionsED import test, train_loop

import torch
from torch import nn
import torch.optim as optim
import pickle
import numpy
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot
import time
import os

# numpy.random.seed(42)

# Bearing data, classic features
PATH_or = os.getcwd()
PATH = os.chdir("..")
PATH = os.getcwd()
PATH = PATH + '\\_Data\\featuresPy_classic.pkl'

os.chdir(PATH_or)

class BearingDataset(Dataset):
    def __init__(self, path, train_size):
        file = open(path, 'rb')
        data = pickle.load(file)
        l = len(data[1])
        l = int(numpy.floor(l*train_size))
        features = data[0]
        labels = data[1]
        features, labels = shuffle(features, labels)
        self.features = features[0:l]
        self.labels = labels[0:l]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
             


hyperparameter = {'numberHidLayer': 3,
                      'layerSize0':54,
                      'layerSize1':52,
                      'layerSize2':41,
                      'dropout0':0.15, 
                      'dropout1':0.15,
                      'dropout2':0.15}

inputFeatures = 10
outputClasses = 4
activationFunction = 'relu'
dataTask = 'Classification'

model = MLP(inputFeatures, outputClasses, activationFunction, dataTask, hyperparameter=hyperparameter)    
print(model)
steps_dataset = 100
steps_training = 100
epochs = 200
g = steps_dataset*steps_training

k_0 = 0

accuracy_list = numpy.zeros([steps_training, steps_dataset])

t0 = time.time()

for k in range(1, steps_dataset+1):
    
    if k != k_0:
        k_0 = k
    
    data = BearingDataset(PATH, train_size=k/100)  
      
    encoder = LabelEncoder()
    encoder.fit(data.labels)
    data.labels = encoder.transform(data.labels)
    
    data.labels = torch.tensor(data.labels, dtype=torch.long)
    data.features = torch.tensor(data.features, dtype=torch.float)
    
    trainX, testX, trainY, testY = train_test_split(data.features, data.labels, train_size=0.8, shuffle=False)
    
    batch_size_train = 45
    batch_size_test = len(testY)
    
    
    scaler = MinMaxScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    
    train_data = []
    test_data = []
    
    for i in range(0, len(trainY)):
        train_data.append((torch.tensor(trainX[i],dtype=torch.float32), trainY[i]))
    for i in range(0, len(testY)):
        test_data.append((torch.tensor(testX[i],dtype=torch.float32), testY[i]))
        
    train_dataloader = DataLoader(train_data, batch_size = batch_size_train, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size_test, shuffle=True)
    
    #%% Model training
    learning_rate = 1e-3
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    accuracy = []
    for i in range(steps_training):
        model = MLP(inputFeatures, outputClasses, activationFunction, dataTask, hyperparameter=hyperparameter)   
        learning_rate = 1e-3
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        accuracy_ep = []
        for t in range(epochs):
            # print(f"Epoch {t+1}\n------------------------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, batch_size_train)
            accuracy_ep.append([t, test(test_dataloader, model, loss_fn)[0],test(test_dataloader, model, loss_fn)[1]])
        # accuracy_ep = numpy.array(accuracy_ep)
        # pyplot.plot(accuracy_ep[:,0],accuracy_ep[:,2])
        accuracy_list[i, k-1] = test(test_dataloader, model, loss_fn)[0]
        print(f"Step\t{(k-1)*steps_training + i+1}/{g}\t{((k-1)*steps_training + i+1)*100/g:>.4f} %\tTime: {(time.time() - t0):>.2f} s")
    # pyplot.show()
    
    # print('Mean accuracy:' + str(numpy.mean(accuracy)))
    # print('Minimum accuracy:' + str(numpy.min(accuracy)))
    # print('Maximum accuracy: ' + str(numpy.max(accuracy)))
    
    t = time.time() - t0
#%% Saving the results
PATH = os.chdir("..")
PATH = os.getcwd()
PATH = PATH + '\\_generatedData'
os.chdir(PATH)
with open('array.npz', 'wb') as file:
    numpy.savez(file, accuracy_over_size = accuracy_list)

os.chdir(PATH_or)
    