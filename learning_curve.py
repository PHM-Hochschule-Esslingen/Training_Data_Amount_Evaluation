# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:00:58 2025

@author: MNeu
"""

import os
import numpy as np
from matplotlib import pyplot
from matplotlib.pyplot import plot
from scipy.optimize import curve_fit
import datetime

from _functionsED import power_law, logarithmic, algebraic_root, arctan_model, exponential_saturation

os.chdir('C:\\Users\\MBraig\\Desktop\\EnoughData\\ED_vEnv1\\_Code')
PATH_or = os.getcwd()

os.chdir('..')

with open(r'_generatedData\2025_08_28_11_51.npz', 'rb') as file:    
    npz = np.load(file)
    data = npz[list(npz)[0]]

plot(data[:, 0])
pyplot.show()

power_law_param = np.zeros((len(data[0, :]), len(data[:, 0]), 3))
power_law_values = np.zeros((len(data[0, :]), len(data[:, 0]), 300))
log_param = np.zeros((len(data[0, :]), len(data[:, 0]), 2))
log_values = np.zeros((len(data[0, :]), len(data[:, 0]), 300))
algebraic_root_param = np.zeros((len(data[0, :]), len(data[:, 0]), 2))
algebraic_root_values = np.zeros((len(data[0, :]), len(data[:, 0]), 300))
arctan_param = np.zeros((len(data[0, :]), len(data[:, 0]), 3))
arctan_values = np.zeros((len(data[0, :]), len(data[:, 0]), 300))
exponential_param = np.zeros((len(data[0, :]), len(data[:, 0]), 3))
exponential_values = np.zeros((len(data[0, :]), len(data[:, 0]), 300))

count_power_law = 0
count_log = 0
count_algebraic_root = 0
count_arctan = 0
count_exponential = 0

maxFev = 100000000

for j in range(len(data[0,:])):
    power_law_param[j, 2, :] = [-250, 260, -0.0002]
    log_param[j, 2, :] = [-250, 260]
    algebraic_root_param[j, 2, :] = [-250, 260]
    arctan_param[j, 2, :] = [-250, 260, -0.0002]
    exponential_param[j, 2, :] = [1.5, -1, 2]
    for i in range(3, len(data[:, 0])):
        # plot(i+1, data[i, 0], 'o')
        # try:
            
        #     power_law_param[j, i,:], _ = curve_fit(power_law, np.linspace(1, i, i), data[0:i, j], 
        #                                            p0=[power_law_param[j, i-1,0], power_law_param[j, i-1,1], power_law_param[j, i-1,2]], maxfev=maxFev)
        #     # power_law_param[j, i,:], _ = curve_fit(power_law, np.linspace(1, i, i), data[0:i, j], 
        #     #                                        bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=2000)
        #     power_law_values[j, i,:] = power_law(np.linspace(1, 300, 300), power_law_param[j, i,0], power_law_param[j, i,1], power_law_param[j, i,2])
        # except:
        #     power_law_param[j, i,:] = [np.inf, np.inf, np.inf]
        #     count_power_law += 1
        # try:
        #     log_param[j, i,:], _ = curve_fit(logarithmic, np.linspace(1, i, i), data[0:i, j], p0=[log_param[j, i-1,0], log_param[j, i-1,1]], maxfev=maxFev)
        #     log_values[j, i,:] = logarithmic(np.linspace(1, 300, 300), log_param[j, i,0], log_param[j, i,1])
        # except:
        #     log_param[j, i,:] = [np.inf, np.inf]
        #     count_log += 1
        # try:
        #     algebraic_root_param[j, i,:], _ = curve_fit(algebraic_root, np.linspace(1, i, i), data[0:i, j], p0=[algebraic_root_param[j, i-1,0], algebraic_root_param[j, i-1,1]], maxfev=maxFev)
        #     algebraic_root_values[j, i,:] = algebraic_root(np.linspace(1, 300, 300), algebraic_root_param[j, i,0], algebraic_root_param[j, i,1])
        # except:
        #     algebraic_root_param[j, i,:] = [np.inf, np.inf]   
        #     count_algebraic_root += 1
        # try:
        #     arctan_param[j, i,:], _ = curve_fit(arctan_model, np.linspace(1, i, i), data[0:i, j], 
        #                                         p0=[arctan_param[j, i-1,0], arctan_param[j, i-1,1], arctan_param[j, i-1,2]+1e-6], maxfev=maxFev)
        #     arctan_values[j, i,:] = arctan_model(np.linspace(1, 300, 300), arctan_param[j, i,0], arctan_param[j, i,1], arctan_param[j, i,2])
        # except:
        #     arctan_param[j, i,:] = [np.inf, np.inf, np.inf]
        #     count_arctan += 1
        try:
            exponential_param[j, i,:], _ = curve_fit(exponential_saturation, np.linspace(1, i, i), data[0:i, j], 
                                                     p0=[exponential_param[j, i-1,0], exponential_param[j, i-1,1], exponential_param[j, i-1,2]], maxfev=maxFev)
            exponential_values[j, i,:] = exponential_saturation(np.linspace(1, 300, 300), exponential_param[j, i,0], exponential_param[j, i,1], exponential_param[j, i,2])
            # plot(data[0:i, j], 'bo')
            # plot(exponential_values[j, i,:], 'r')
            # pyplot.show()
            # print(exponential_param[j, i,:])
        except:
            exponential_param[j, i,:] = [np.inf, np.inf, np.inf]
            count_exponential += 1
    print(str(j) + f'/{len(data[0,:])}')
    
pyplot.show()


#%% Saving the results
# PATH = os.chdir("..")
PATH = os.getcwd()
PATH = PATH + '\\_generatedData'
os.chdir(PATH)
# string = 'result_learning_curve_power_law'
# with open(f'{string}.npz', 'wb') as file:
#     np.savez(file, param = power_law_param, values = power_law_values)
# string = 'result_learning_curve_log'
# with open(f'{string}.npz', 'wb') as file:
#     np.savez(file, param = log_param, values = log_values)
# string = 'result_learning_curve_algebraic_root'
# with open(f'{string}.npz', 'wb') as file:
#     np.savez(file, param = algebraic_root_param, values = algebraic_root_values)
# string = 'result_learning_curve_arctan'
# with open(f'{string}.npz', 'wb') as file:
#     np.savez(file, param = arctan_param, values = arctan_values)
string = 'result_learning_curve_exponential_1'
with open(f'{string}.npz', 'wb') as file:
     np.savez(file, param = exponential_param, values = exponential_values)
os.chdir(PATH_or)