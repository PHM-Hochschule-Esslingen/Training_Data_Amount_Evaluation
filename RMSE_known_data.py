# -*- coding: utf-8 -*-
"""
Spyder-Editor

Dies ist eine temporäre Skriptdatei.

"""
from _functionsED import logarithmic, logarithmic_inv, algebraic_root, algebraic_root_inv, power_law, power_law_inv, arctan_model, arctan_model_inv, exponential_saturation, exponential_saturation_inv
from _functionsED import plot_tangent
import numpy as np
from matplotlib import pyplot
import os
import time 
from scipy.io import savemat

#%% Load accuracy data
os.chdir('..')
with open(r'_generatedData\2025_08_28_11_51.npz', 'rb') as file:    
    npz = np.load(file)
    accs = npz[list(npz)[0]]

#%% load learning curve experiment results
with open("C:/Users/MBraig/Desktop/EnoughData/ED_vEnv1/_generatedData/result_learning_curve_exponential_1.npz", 'rb') as file:
    data = np.load(file)
    
    param = data['param']
    values = data['values']
    
#%% rmse evaluation
rmse = np.zeros((len(values[:, 0, 0]), 100))
rmse_5 = np.zeros(97)
rmse_50 = np.zeros(97)
rmse_95 = np.zeros(97)
for j in range(0, len(values[:, 0, 0])): 
    no = j
    for i in range(3, 100):
        known = i
        
        rmse[j, i] = np.sqrt(np.sum((accs[:known, no] - values[no, known, :known])**2)/known)

rmse = rmse[:, 3:]

for k in range(len(rmse[0, :])):
    rmse_5[k] = np.quantile(rmse[:, k], 0.05)
    rmse_50[k] = np.quantile(rmse[:, k], 0.5)
    rmse_95[k] = np.quantile(rmse[:, k], 0.95)

x = np.linspace(4, 100, 97)

result = np.vstack((x, rmse_5, rmse_50, rmse_95))
#%% save results
mdic = {'result': result}
savemat(r'_generatedData\rmse_known_exponential_2.mat', mdic)
