# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:23:40 2025

@author: MBraig
"""

from Nets_Only_Pytorch import MLP
from functionsED import test, train_loop

import torch
from torch import nn, device
import torch.optim as optim
import pickle
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot
import time
from datetime import datetime

from multiprocessing import Process, Queue, Pipe
import logging

import os

# Konfiguration des Log-Handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(processName)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# check if gpu hardware is available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = 'cuda:0'
    # torch.set_default_device('cuda')
else:
    device = 'cpu'
    print("No GPU available. Training will run on CPU.")

device = 'cpu'
# np.random.seed(42)

# Bearing data, classic features
PATH_or = os.getcwd()
PATH = PATH_or[:-6] + '\\_data\\featuresPy_classic.pkl'

os.chdir(PATH_or)


# define dataset class 
class BearingDataset(Dataset):
    def __init__(self, path, indices):
        # open file and load data
        file = open(path, 'rb')
        data = pickle.load(file)
        
        features = data[0]
        labels = data[1]
        
        features = features[indices]
        labels = labels[indices]
        
        self.features = features
        self.labels = labels
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
              
# load shuffled indices lists
with open(PATH_or[:-6] + '/_generatedData/indicesList.npz', 'rb') as file:
    indices = np.load(file)['arr_0']

indices = np.int16(indices)

def MLP_Training(indices, q, conn):
    # set hyperparameter fot the MLP
    hyperparameter = {'numberHidLayer': 3,
                          'layerSize0':50,
                          'layerSize1':50,
                          'layerSize2':40,
                          'dropout0':0.05, 
                          'dropout1':0.05,
                          'dropout2':0.05}
    
    inputFeatures = 10
    outputClasses = 4
    activationFunction = 'relu'
    dataTask = 'Classification'
    
    # Instantiate Model and pass to GPU
    model = MLP(inputFeatures, outputClasses, activationFunction, dataTask, hyperparameter=hyperparameter)
    
    # print model architecture 
    print(model)
    
    # set steps in dataset size (number of equally sized steps to 100 %)
    steps_dataset = 100
    stepsize_dataset = int(100/steps_dataset)
    
    
    # number of training epochs 
    epochs = 100
    
    # number of trainings (used for progress indication in terminal)
    g = steps_dataset*len(indices[0, :])
                                         
    # predefining array for results
    accuracy_list = np.zeros([steps_dataset, len(indices[0, :])])
    
    # start time stopping 
    t0 = time.time()
    
    # counters
    g_i = 0
    
    
    #%% Loop for different shuffled datasets
    for i in range(0, len(indices[0, :])):
        
        # loading the data, shuffled after the indices in the indices-array
        data = BearingDataset(PATH, indices[:, i])
        
        # encode labels 
        encoder = LabelEncoder()
        encoder.fit(data.labels)
        data.labels = encoder.transform(data.labels)
        
        # split the dataset - 75 % train data, 25 % test data - without shufffling again
        trainX, testX, trainY, testY = train_test_split(data.features, data.labels, train_size=0.75, shuffle=False)
        
        # prepare data for training
        test_data = []
            
        
    
        #%% Loop for different dataset sizes - 1 ... 100 %
        c = 0
        for j in range(1, 101, stepsize_dataset):
            
            trainX_j = trainX[:int(np.floor(len(trainY)*j/100)), :]
            trainY_j = trainY[:int(np.floor(len(trainY)*j/100))]
            
            # calculate batch size 
            batch_size_train = int(np.ceil(len(trainY_j)/10))
            batch_size_test = len(testY)
            
            # scale feature values from 0 to 1 -> fit scaler only on training data
            scaler = MinMaxScaler()
            scaler.fit(trainX_j)
            trainX_j = scaler.transform(trainX_j)
            testX_j = scaler.transform(testX)
                    
            train_data = []
            for k in range(0, len(trainY_j)):
                train_data.append((torch.tensor(trainX_j[k],dtype=torch.float32, device=device), torch.tensor(trainY_j[k], device=device)))
            for k in range(0, len(testY)):
                test_data.append((torch.tensor(testX_j[k],dtype=torch.float32, device=device), torch.tensor(testY[k], device=device)))
                
            train_dataloader = DataLoader(train_data, batch_size = batch_size_train, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size = batch_size_test, shuffle=True)
            
            #%% Model training
            # select learning rate 
            learning_rate = 1e-3
            
            # select loss funtion for classification
            loss_fn = nn.CrossEntropyLoss()
            
            # select optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            accuracy = []
            
            # reinstantiate model to reset model weights to random values
            model = MLP(inputFeatures, outputClasses, activationFunction, dataTask, hyperparameter=hyperparameter, device=device)
            
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            accuracy_ep = []
            
            # train for the defined amount of epochs 
            for t in range(epochs):
                train_loop(train_dataloader, model, loss_fn, optimizer, batch_size_train)
                accuracy_ep.append([t, test(test_dataloader, model, loss_fn)[0],test(test_dataloader, model, loss_fn)[1]])
            
            # save result of every training
            accuracy_list[c, i] = test(test_dataloader, model, loss_fn)[0]
            c += 1 
            g_i += 1
            
            # print prgress to terminal
            logger.info(f"Step\t{g_i}/{g}\t{g_i*100/g:>.4f} %\tTime: {(time.time() - t0):>.2f} s")
            
            # stop time 
            t = time.time() - t0

    q.put(accuracy_list)
    
    
if __name__ =="__main__":
    data = np.array([])
    processes = []
    queues = []
    
    for i in range(6):
        parent_conn, child_conn = Pipe()
        q = Queue()
        dataOffset = 0
        dataPerProcess = 2000
        p = Process(target=MLP_Training, args=(indices[:, dataOffset + dataPerProcess*i:dataOffset + (i+1)*dataPerProcess], q, child_conn,), name=f"Prozess-{i}")
        processes.append(p)
        queues.append(q)
        p.start()
    
    for q in queues:
        if data.size == 0:
            data = np.array(q.get())
        else:
            data = np.hstack((data, np.array(q.get())))
    
    for p in processes:
        p.join()
        

    
    

#%% Saving the results
PATH = os.chdir("..")
PATH = os.getcwd()
PATH = PATH + '\\_generatedData'
os.chdir(PATH)
string = datetime.now().strftime('%Y_%m_%d_%H_%M')
with open(f'{string}.npz', 'wb') as file:
    np.savez(file, accuracy_over_size = data)
os.chdir(PATH_or)
