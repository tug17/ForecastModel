#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sebastian Gegenleithner & Manuel Pirker
"""

#############################
#         Imports
#############################
import numpy as np
import os
import json

#############################
#         Functions
#############################
# helper functions
def get_n_peaks(df, col_eval, n_peaks, window):
    peaks = []
    df_length = df.shape[0]
    for n_peak in range(n_peaks):
        i_max   = df[col_eval].argmax()
        i_start = np.max([0, i_max - window//2])
        i_end   = np.min([df_length, i_max + window//2])
        
        peaks.append(df.iloc[i_start:i_end])
        
        df = df.drop(df.index[i_start:i_end], axis=0)
    
    return peaks

def extract_arima_metrics(path):
    with open(os.path.join(path, "model.json"),"r") as f:
        dic = json.load(f)
        
    for metric in ["NSE", "KGE", "bias"]:
        array = []
        for n_fold in range(1,6):
            array.append(dic[f"Fold_{n_fold:d}"][metric])
            
        np.savetxt(os.path.join(path, f"metric_{metric.lower()}.txt"),
                      np.array(array), delimiter=",")

    return dic

def regroup_metrics(path):
    metric = {"eval": {},
             "test": {}}
    for key in ["nse", "kge", "bias"]:
        metric["test"][key] = np.loadtxt(os.path.join(path, f"metric_{key}.txt"), delimiter=",").tolist()

    with open(os.path.join(path, "metrics.txt"), "w+") as f:
        json.dump(metric, f)  

def find_best_model(directory, metric_key="kge"):
    valid = []
    test = []
    for path in os.listdir(directory):
        try:
            with open(os.path.join(directory, path, "metrics.txt"), "r") as f:
                metrics = json.load(f)
        except:
            break
        valid.append(np.mean(metrics["valid"][metric_key][2:]))
        test.append(np.mean(metrics["test"][metric_key]))
        
    return valid, test

def eval_peak_distance(time,meas_data,sim_data,hq1,window):
    
    # find indices in meas data larger than hq1
    indices = np.where(meas_data > hq1)
    indices = indices[0].tolist()
    indices.append(1)
    
    chunks = []
    chunk = []
    
    for i in range(0,len(indices)-1):
       # print(indices[0][i])
        if len(chunk) == 0:
            chunk.append(indices[i])
        if abs(indices[i] - indices[i+1]) < window:
            chunk.append(indices[i+1])
        else:
            chunks.append(chunk)
            chunk = []
    
    # reinit array blow up chuncks to measurement array
    chuncks_blown = []
    
    for chunk in chunks:
        min_val = min(chunk) - window
        max_val = max(chunk) + window
        
        range_new = list(range(min_val, max_val))
        
        chuncks_blown.append(range_new)
    
    # compute distances
    times = []
    vals_meas = []
    vals_sim = []
    
    x_sim = []
    y_sim = []
    x_meas = []
    y_meas = []
    
    for chunck in chuncks_blown:
        
        val_meas_i = np.asarray(meas_data[chunck])
        val_sim_i = np.asarray(sim_data[chunck])
        
        # normalize
        time_norm = time[chunck]
        
        times.append(time_norm)
        vals_meas.append(val_meas_i)
        vals_sim.append(val_sim_i)
    
    return times, vals_meas, vals_sim, chuncks_blown, x_sim,y_sim,x_meas,y_meas

# evaluate over forecasting horizont
def evaluate_multistep(obs_multistep, pred_multistep, loss_function):
    # print((obs_multistep.shape), pred_multistep.shape)
    if obs_multistep.shape[1] == pred_multistep.shape[1]:
        step_losses = [loss_function(obs_multistep[:,x,0], pred_multistep[:,x]) 
                       for x in range(pred_multistep.shape[1])] 
    else:
        step_losses = [loss_function(obs_multistep[:,0], pred_multistep[:,x]) 
                       for x in range(pred_multistep.shape[1])] 

    return step_losses

#%% metrics
def calculate_rms(observed, predicted):
    return np.sqrt(np.mean((observed - predicted)**2))

def calculate_nse(observations, predictions):
    nse = (1 - ((predictions-observations)**2).sum() / ((observations-observations.mean())**2).sum())
    return nse

def calculate_kge(observations, predictions):
    
    m1, m2 = np.nanmax((np.nanmean(observations), 1e-6)), np.nanmax((np.nanmean(predictions), 1e-6))
    r = np.sum((observations - m1) * (predictions - m2)) / (np.sqrt(np.sum((observations - m1) ** 2)) * np.sqrt(np.sum((predictions - m2) ** 2)))
    
    beta = m2 / m1
    gamma = (np.std(predictions) / m2) / (np.std(observations) / m1)
    
    # alpha = np.std(predictions) / np.std(observations)
    
    KGE =  1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    
    return KGE
    
def calculate_kge5alpha(observations, predictions):
    
    m1, m2 = np.nanmax((np.nanmean(observations), 1e-6)), np.nanmax((np.nanmean(predictions), 1e-6))
    r = np.sum((observations - m1) * (predictions - m2)) / (np.sqrt(np.sum((observations - m1) ** 2)) * np.sqrt(np.sum((predictions - m2) ** 2)))
    
    beta = m2 / m1
    gamma = (np.std(predictions) / m2) / (np.std(observations) / m1)
    
    alpha = np.std(predictions) / np.std(observations)
    
    KGE =  1 - np.sqrt((r - 1) ** 2 + (2*(alpha - 1)) ** 2 + (beta - 1) ** 2)
    
    return KGE

def calculate_bias(observations, predictions):

    numerator   = np.sum(observations - predictions)
    denominator = np.sum(observations)
    
    pbias = (numerator / denominator) * 100
    
    return pbias