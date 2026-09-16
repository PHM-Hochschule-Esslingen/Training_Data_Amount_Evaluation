# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:46:44 2025

@author: MNeu

-------------------------------------------------------------------------------
NOTE
-------------------------------------------------------------------------------
By construction, empDensity creates two empty boundary bins
(dens = [0]*(n_classes+2); only the indices 1..n_classes are filled).
Therefore len(dens) = L + 2 with L = n_classes, and the entropy normalisation
according to Eq. (15)/(16) has to be carried out with log2(len(dens) - 2).
-------------------------------------------------------------------------------
"""
#%% Import packages
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot
import pickle


#%% Function movingAvg 
# Author: MN

# Input: 
#   arr: Array for calculating moving average
#   m: number of summands per point (backwards)

# Output:
#   array with the applied moving average
def movingAvg(arr, m):
    arr_avg = arr.copy()
    for i in range(0, len(arr)):
        if i < m and i != 0:           
            arr_avg[i] = sum(arr[0:i+1])/(i+1)
        elif i >= m:
            arr_avg[i] = sum(arr[(i+1-m):i+1])/m
    return arr_avg

#%% Function solOdeFirstOrder
def solOdeFirstOrder(x, a, b, c):
    return a-(a-b)*np.exp(-c*x)

#%% Function Entropy
# Author: MN

# Input:
#   p: empirical density function to calculate the entropy of
# Output:
#   H: entropy of the density function p
def Entropy(p):
    H = 0 
    for p_i in p:
        if p_i != 0:
            H = H - p_i*np.log2(p_i)
        else:
            None
    return H

#%% Function KLDivergence
# Author: MN

# Input:
#   p, q: two empirical density functions
# Output:
#   KL: KL-Divergence of the two density functions p, q
def KLDivergence(p, q):
    KL = 0
    for i in range(0, len(p)):
        if p[i] != 0 and q[i] != 0:
            KL = KL + p[i]*np.log(p[i]/q[i])
        else: 
            None
    return float(KL)

#%% Function empDensity
# Author: MN

# Input:
#   data: data of which the emprirical density function should be calculated
#   manualbins: (optional) set to True if the number of bins should be set in the function call
#   bins: number of bins
#   manualrange: (optional) set to True to impose the bin edges of a
#                reference dataset instead of deriving them from min/max of data
#   minf_m, maxf_m: lower/upper edge of the imposed range
def empDensity(data, manualbins=False, bins=2, manualx=False, x_m = [0,1],
               manualrange=False, minf_m=0.0, maxf_m=1.0):
    if manualbins == True:
        n_classes = bins-2
    elif manualbins == False:
        n_classes = int(np.floor(np.sqrt(len(data))))
    
    if manualrange == True:
        minf = minf_m
        maxf = maxf_m
    else:
        minf = min(data)
        maxf = max(data)
    width = (maxf - minf)/n_classes
    
        
    data = sorted(data)
    

    if manualx == False:  
        dens = [0]*(n_classes+2)
        x = [0]*(n_classes+2)
        for i in range(1, n_classes+1):
            x[i] = minf + width/2 + (i-1)*width
        x[0] = x[1]-width
        x[-1] = x[-2]+width
    else:
        dens = [0]*len(x_m)
        x = [0]*len(x_m)
        x = x_m

    for i in range(1, n_classes+1):
        if i == 1:
            k = 0
            while k <= len(data)-1 and data[k] < minf + width:
                k = k + 1
                dens[i] = dens[i] + 1
        elif i == n_classes:
            while k <= len(data)-1 and data[k] >= maxf - width:
                k = k + 1
                dens[i] = dens[i] + 1
        else:
            while k <= len(data)-1 and data[k] >= minf + (i-1)*width and data[k] < minf + (i)*width:
                k = k + 1
                dens[i] = dens[i] + 1

    dens = np.array(dens)/len(data)
    return x, dens

#%% Trainloop and Testfunctions used from pytorch.org
def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return [correct, test_loss]

#%% Model functions
def power_law(x, a, b, c):
    # Author: MF

    return (1 - a) - b * np.power(x, c)

def power_law_inv(y, a, b, c):
    # Author: MN

    return (np.power((1-a-y)/b, 1/c), y)

def logarithmic(x, a, b):
    # Author: MF

    return a + b * np.log(x)

def logarithmic_inv(y, a, b):
    # Author: MN

    return (np.e**((y-a)/b), y)

def algebraic_root(x, a, b):
    # Author: MF

    return a + b * np.sqrt(x)

def algebraic_root_inv(y, a, b):
    # Author: MN

    return (((y-a)/b)**2, y)

def arctan_model(x, a, b, c):
    # Author: MF

    return a + b * np.arctan(c * x)

def arctan_model_inv(y, a, b, c):
    # Author: MN
    if a+b*np.pi/2 > y:
        return ((np.tan((y-a)/b))/c, y)
    else: 
        return (np.sqrt(b/(0.01*c)-1/c**2), arctan_model(np.sqrt(b*c/(0.01*c**2)-1/c**2), a, b, c))

def exponential_saturation(x, a, b, c):
    # Author: MF

    return a - (a - b) * np.exp(-c * x)

def exponential_saturation_inv(y, a, b, c):
    # Author: MN
    if a > y: 
        return (-(np.log(-(y-a)/(a-b)))/c, y)
    else:
        return (-(np.log(0.01/(c*(a-b))))/c, exponential_saturation(-(np.log(0.01/(c*(a-b))))/c, a, b, c))
        
#%% Fit extension functions
def extend_fit(fit_func, params, x_start, x_end=1800, steps=1000):
    # Author: MF

    x_extended = np.linspace(x_start, x_end, steps)
    y_extended = fit_func(x_extended, *params)
    return x_extended, y_extended

def find_threshold_x(fit_func, params, threshold=0.8, x_vals=None):
    # Author: MF

    if x_vals is None:
        x_vals = np.linspace(0, 2000, 10000)
    y_vals = fit_func(x_vals, *params)
    above = np.where(y_vals >= threshold)[0]
    if len(above) == 0:
        return None
    return x_vals[above[0]]

def conservative_cutoff(fit_func, params, x_vals=None, threshold_ratio=0.99):
    # Author: MF

    if x_vals is None:
        x_vals = np.linspace(0, 2000, 10000)
    y_vals = fit_func(x_vals, *params)
    y_max = np.max(y_vals)
    target_y = y_max * threshold_ratio
    close_idx = np.where(y_vals >= target_y)[0]
    if len(close_idx) == 0:
        return None
    return x_vals[close_idx[0]]

def plot_tangent(x, y, m, x_range):
    print(f'Slope of the tangent: {m}')
    return (x_range, y + m*(x_range - x))
