# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:23:40 2025

@author: MBraig
"""

from _functionsED import logarithmic, logarithmic_inv, algebraic_root, algebraic_root_inv, power_law, power_law_inv, arctan_model, arctan_model_inv, exponential_saturation, exponential_saturation_inv
from _functionsED import plot_tangent
from _functionsED import Entropy, empDensity, movingAvg
import numpy as np
from matplotlib import pyplot
import os
import time 
from torch.utils.data import Dataset
import pickle
from sklearn.metrics import mean_squared_error
from scipy.io import savemat

#%% load and prepare data, calculate gne
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


#%% paths - adapt to the local directory structure before running
# ACCURACY_FILE: .npz archive containing the learning curves. The first array of
#                the archive is used; expected shape (N_P, N_rep), i.e. one row
#                per relative subset size p and one column per repetition.
# FEATURE_FILE:  pickled dataset (features, labels) that BearingDataset reads.
# GENERATED_DIR: directory holding 'indicesList.npz' (the shuffled index lists)
#                and receiving the resulting .mat file.
ACCURACY_FILE = 'ENTER PATH TO ACCURACY FILE (.npz)'
FEATURE_FILE  = 'ENTER PATH TO FEATURE FILE (.pkl)'
GENERATED_DIR = 'ENTER PATH TO GENERATED DATA DIRECTORY'

with open(ACCURACY_FILE, 'rb') as file:    
    npz = np.load(file)
    accs = npz[list(npz)[0]]


path = FEATURE_FILE

# define .mat saving dictionary
mdic = {}

# load shuffled indices lists
with open(os.path.join(GENERATED_DIR, 'indicesList.npz'), 'rb') as file:
    indices = np.load(file)['arr_0']
indices = np.int16(indices)

h_p    = 0.01
o      = 10
N_TRAIN = 450
N_P     = 100
N_rep  = len(indices[0, :12000])

p_grid = np.arange(1, N_P+1)/N_P

featureData = np.zeros((N_TRAIN, N_rep))
entropy = np.zeros((N_P, N_rep))
gne = np.full((N_P, N_rep), np.nan)
mse = np.zeros((N_P, N_rep))
corr = np.zeros((3, 3, N_rep))

for i in range(0, N_rep):
    featureData[:, i] = BearingDataset(path, indices[:, i]).features[:N_TRAIN, 3]
    xFull, densFull = empDensity(featureData[:, i])
    minf_full = min(featureData[:, i])
    maxf_full = max(featureData[:, i])
    for j in range(1, N_P+1): 
        xPart, densPart = empDensity(featureData[:int(len(featureData[:, i])*j/N_P), i])

        L = len(densPart) - 2
        entropy[j-1, i] = Entropy(densPart)/np.log2(L)

        xPart, densPart = empDensity(featureData[:int(len(featureData[:, i])*j/N_P), i],
                                     manualbins=True, bins=int(len(densFull)),
                                     manualx=True, x_m=xFull,
                                     manualrange=True, minf_m=minf_full, maxf_m=maxf_full)
        mse[j-1, i] = mean_squared_error(densFull, densPart)

    entropy[:, i] = movingAvg(entropy[:, i], m=o)
    gne[1:, i] = np.abs(np.diff(entropy[:, i]))/h_p

    corr[:, :, i] = np.corrcoef(np.vstack((mse[1:, i], accs[1:, i], gne[1:, i])))

    print(f'Step: {i}/{N_rep}')
    
#%%
fig, ax1 = pyplot.subplots()
x, y = empDensity(corr[1, 0, :]) # density corrcoef between mse and accuracies
mdic['corr_mse_accs']=corr[1, 0, :]
mdic['dens_corr_mse_accs'] = np.hstack((np.reshape(x, shape=(len(x), 1)),np.reshape(y, shape=(len(y), 1))))
print(np.sum(y))
ax1.plot(x, y, label='mse and accs')
ax1.fill_between(x[:max(np.where(np.cumsum(y) < 0.95)[0])], 
                 y[:max(np.where(np.cumsum(y) < 0.95)[0])], 
                 np.zeros(len(x[:max(np.where(np.cumsum(y) < 0.95)[0])])), 
                 color='C0', alpha=0.5)
x, y = empDensity(corr[2, 0, :]) # density corrcoef between mse and grad
mdic['corr_mse_grad']=corr[2, 0, :]
mdic['dens_corr_mse_grad'] = np.hstack((np.reshape(x, shape=(len(x), 1)),np.reshape(y, shape=(len(y), 1))))
print(np.sum(y))
ax1.plot(x, y, label='mse and gne')
ax1.fill_between(x[min(np.where(np.cumsum(y) > 0.05)[0]):], 
                 y[min(np.where(np.cumsum(y) > 0.05)[0]):], 
                 np.zeros(len(x[min(np.where(np.cumsum(y) > 0.05)[0]):])), 
                 color='C1', alpha=0.5)
x, y = empDensity(corr[2, 1, :]) # density corrcoef between accs and grad
mdic['corr_grad_accs']=corr[2, 1, :]
mdic['dens_corr_grad_accs'] = np.hstack((np.reshape(x, shape=(len(x), 1)),np.reshape(y, shape=(len(y), 1))))
print(np.sum(y))
ax1.plot(x, y, label='accs and gne')
ax1.fill_between(x[:max(np.where(np.cumsum(y) < 0.95)[0])], 
                 y[:max(np.where(np.cumsum(y) < 0.95)[0])], 
                 np.zeros(len(x[:max(np.where(np.cumsum(y) < 0.95)[0])])), 
                 color='C2', alpha=0.5)
ax1.set_xlim([-1, 1])
ax1.legend(ncols=3)
ax1.grid()
pyplot.show()

gne_5  = np.full(N_P, np.nan)
gne_50 = np.full(N_P, np.nan)
gne_95 = np.full(N_P, np.nan)

for k in range(1, N_P):
    gne_5[k]  = np.quantile(gne[k, :], 0.05)
    gne_50[k] = np.quantile(gne[k, :], 0.5)
    gne_95[k] = np.quantile(gne[k, :], 0.95)

pyplot.fill_between(p_grid[1:], gne_5[1:], gne_95[1:], color='C0', alpha=0.5)
pyplot.plot(p_grid[1:], gne_50[1:])
pyplot.xlabel('Relative training subset size p')
pyplot.ylabel('GNE(p)')
pyplot.show()

#%% gne vs acc evaluation
steps = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

idx_sat    = np.full((len(steps), N_rep), N_P-1, dtype=int)
p_sat      = np.zeros((len(steps), N_rep))
acc        = np.zeros((len(steps), N_rep))
delta_acc  = np.zeros((len(steps), N_rep))
terminated = np.zeros((len(steps), N_rep), dtype=bool)

for i in range(N_rep):
    acc_max = np.median(accs[-20:, i])
    for j, t in enumerate(steps):
        for k in range(2, N_P):
            if gne[k, i] <= t*np.max(gne[1:k+1, i]):
                idx_sat[j, i]    = k
                terminated[j, i] = True
                break
        p_sat[j, i]     = p_grid[idx_sat[j, i]]
        acc[j, i]       = accs[idx_sat[j, i], i]
        delta_acc[j, i] = acc_max - acc[j, i]

count_not_terminated = (~terminated).sum(axis=1)

pyplot.pcolormesh(acc)
pyplot.show()

fig, ax = pyplot.subplots()
for i in range(len(steps)):
    x1, y1 = empDensity(p_sat[i, :], manualbins=True, bins=33)
    ax.plot(x1, y1, label=f't = {steps[i]}')
ax.set_xlabel(r'$p_{i,sat}$')
ax.set_ylabel('Relative frequency')
ax.legend()
pyplot.show()

#%%
fig, ax = pyplot.subplots(2)

bplot = ax[0].boxplot([acc[i, :] for i in range(len(acc[:,0]))], positions=range(1, len(acc[:,0])+1), 
           tick_labels=np.round(steps, 3), whis=(5, 95), showfliers=False, patch_artist=True)
ax[0].set_xlabel('stop criterion / GNE')
ax[0].set_ylabel('V(n) / acc')

# fill with colors
for patch in bplot['boxes']:
    patch.set_facecolor('white')

bplot = ax[1].boxplot([delta_acc[i, :] for i in range(len(acc[:,0]))], positions=range(1, len(acc[:,0])+1), 
           tick_labels=np.round(steps, 3), whis=(5, 95), showfliers=False, patch_artist=True)
ax[1].set_xlabel('stop criterion / GNE')
ax[1].set_ylabel(r'Delta_Acc')

# fill with colors
for patch in bplot['boxes']:
    patch.set_facecolor('white')
ax[0].grid()
ax[1].grid()
pyplot.show()
#%% saving results
mdic['gne']                     = gne
mdic['mse']                     = mse
mdic['p_grid']                  = p_grid
mdic['grad_quantiles']          = np.vstack((p_grid[1:], gne_5[1:], gne_50[1:], gne_95[1:]))
mdic['gne_stop_index']          = idx_sat
mdic['p_isat']                  = p_sat
mdic['gne_step']                = steps
mdic['gne_accs']                = acc
mdic['delta_acc']               = delta_acc
mdic['terminated']              = terminated
mdic['counter_not_terminated']  = count_not_terminated
mdic['smoothing_order']         = o
mdic['h_p']                     = h_p

savemat(os.path.join(GENERATED_DIR, f'result_gne_smoothing_{o}.mat'), mdic)
