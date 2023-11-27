#!/usr/bin/env python3
import numpy as np

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

def calculate_rms(observed, predicted):
    return np.sqrt(mean_squared_error(observed, predicted))

def calculate_nse(observations, predictions):
    nse = (1 - ((predictions-observations)**2).sum() / ((observations-observations.mean())**2).sum())
    return nse

def calculate_kge(observations, predictions):
    
    m1, m2 = np.mean(observations), np.mean(predictions)
    r = np.sum((observations - m1) * (predictions - m2)) / (np.sqrt(np.sum((observations - m1) ** 2)) * np.sqrt(np.sum((predictions - m2) ** 2)))
    beta = m2 / m1
    gamma = (np.std(predictions) / m2) / (np.std(observations) / m1)
    
    # alpha = np.std(predictions) / np.std(observations)
    
    KGE =  1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    
    return KGE

def calculate_bias(observations, predictions):
    pbias = np.sum((observations - predictions) / observations) * 100 / len(observations)
    return pbias

    
def evaluate_multistep(obs_multistep, pred_multistep, loss_function):
    # print((obs_multistep.shape), pred_multistep.shape)
    if obs_multistep.shape[1] == pred_multistep.shape[1]:
        step_losses = [loss_function(obs_multistep[:,x,0], pred_multistep[:,x]) 
                       for x in range(pred_multistep.shape[1])] 
    else:
        step_losses = [loss_function(obs_multistep[:,0], pred_multistep[:,x]) 
                       for x in range(pred_multistep.shape[1])] 

    return step_losses