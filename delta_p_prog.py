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

#%%
os.chdir('..')
with open(r'_generatedData\2025_08_28_11_51.npz', 'rb') as file:    
    npz = np.load(file)
    accs = npz[list(npz)[0]]


# fileDict = {'0':{
#             'source'       :"C:/Users/MBraig/Desktop/EnoughData/ED_vEnv1/_generatedData/result_learning_curve_algebraic_root.npz",
#             'destination'  :'_generatedData\delta_p_algebraic_root_1.mat'},
            
#             '1':{
#             'source'       :"C:/Users/MBraig/Desktop/EnoughData/ED_vEnv1/_generatedData/result_learning_curve_arctan.npz",
#             'destination'  :'_generatedData\delta_p_arctan_1.mat'},
            
#             '2':{
#             'source'       :"C:/Users/MBraig/Desktop/EnoughData/ED_vEnv1/_generatedData/result_learning_curve_exponential.npz",
#             'destination'  :'_generatedData\delta_p_exponential_1.mat'},
            
#             '3':{
#             'source'       :"C:/Users/MBraig/Desktop/EnoughData/ED_vEnv1/_generatedData/result_learning_curve_log.npz",
#             'destination'  :'_generatedData\delta_p_log_1.mat'},
            
#             '4':{
#             'source'       :"C:/Users/MBraig/Desktop/EnoughData/ED_vEnv1/_generatedData/result_learning_curve_power_law.npz",
#             'destination'  :'_generatedData\delta_p_power_law_1.mat'}}

fileDict = {'0':{
            'source'       :"C:/Users/MBraig/Desktop/EnoughData/ED_vEnv1/_generatedData/result_learning_curve_exponential_1.npz",
            'destination'  :'_generatedData\delta_p_exponential_2.mat'}}
#%%
for f in range(len(fileDict)):
    with open(fileDict[str(f)]['source'], 'rb') as file:
        data = np.load(file)
        
        param = data['param']
        values = data['values']
        values = values[:,:,:100]
    #%%
    knownValues = np.array([5, 10, 15]) # number of known accuracy Values for prognosis
    accThreshold = np.array([0.60, 0.65, 0.70, 0.75, 0.80]) # accuracy thresholds
    
    # arrays for results
    delta_p = np.zeros((len(values[:, 0, 0]), len(accThreshold), len(knownValues)))
    delta_p_5  = np.zeros((len(accThreshold), len(knownValues)))
    delta_p_25  = np.zeros((len(accThreshold), len(knownValues)))
    delta_p_50 = np.zeros((len(accThreshold), len(knownValues)))
    delta_p_75  = np.zeros((len(accThreshold), len(knownValues)))
    delta_p_95 = np.zeros((len(accThreshold), len(knownValues)))
    
    # counters
    counter1 = np.zeros((len(accThreshold), len(knownValues))) # real acc doesnt reach thresh
    counter2 = np.zeros((len(accThreshold), len(knownValues))) # acc of fitted curve doesnt reach thresh
    counter3 = np.zeros((len(accThreshold), len(knownValues))) # both doesnt reach thresh
    counter4 = np.zeros((len(accThreshold), len(knownValues))) # real acc have reached thresh before (prognosis=false)
    counter5 = np.zeros((len(accThreshold), len(knownValues))) # ok
    
    for j in range(0, len(values[:, 0, 0])): # learning curves loop
        for i in range(len(knownValues)): # data for prognosis loop 
            for k in range(len(accThreshold)): # threshold loop
                if all(accs[knownValues[i]:,j] < accThreshold[k]) and all(values[j, knownValues[i]-1,:] < accThreshold[k]):
                    counter3[k,i]+=1
                    delta_p[j,k,i]=np.nan
                elif all(accs[knownValues[i]:,j] < accThreshold[k]): # if accuracy from now on doesnt reach threshold
                    counter1[k,i]+=1
                    delta_p[j,k,i]=np.nan
                elif all(values[j, knownValues[i]-1,:] < accThreshold[k]): # if model function from now on doesnt reach threshold
                    counter2[k,i]+=1 
                    delta_p[j,k,i]=np.nan
                elif any(accs[:knownValues[i],j] >= accThreshold[k]):
                    counter4[k,i]+=1 
                    delta_p[j,k,i]=np.nan
                else:
                    counter5[k,i] += 1
                    p1 = min(np.where(accs[:,j]>=accThreshold[k])[0])
                    p2 = min(np.where(values[j,knownValues[i]-1,:]>=accThreshold[k])[0])
                    delta_p[j,k,i]= p1-p2
    #%%
    for i in range(len(knownValues)): # data for prognosis loop 
        for k in range(len(accThreshold)): # threshold loop
            idx = np.where(np.isnan(delta_p[:,k,i])==False)[0]
            if len(idx>0):
                delta_p_5[k,i]  = np.quantile(delta_p[idx,k,i], 0.05)
                delta_p_25[k,i] = np.quantile(delta_p[idx,k,i], 0.25)
                delta_p_50[k,i] = np.quantile(delta_p[idx,k,i], 0.5)
                delta_p_75[k,i] = np.quantile(delta_p[idx,k,i], 0.75)
                delta_p_95[k,i] = np.quantile(delta_p[idx,k,i], 0.95)
            else:
                delta_p_5[k,i]  = np.nan
                delta_p_25[k,i] = np.nan
                delta_p_50[k,i] = np.nan
                delta_p_75[k,i] = np.nan
                delta_p_95[k,i] = np.nan
    #%% save results
    mdic = {'delta_p' : delta_p,
            'delta_p_5': delta_p_5,
            'delta_p_25': delta_p_25,
            'delta_p_50': delta_p_50,
            'delta_p_75': delta_p_75,
            'delta_p_95':delta_p_95,
            'counter1': counter1,
            'counter2': counter2,
            'counter3': counter3,
            'counter4': counter4,
            'counter5': counter5}
    
    savemat(fileDict[str(f)]['destination'], mdic)
    print('x')
