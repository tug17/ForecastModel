import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import pickle
# import json

import os

class CreateIndices:
    def __init__(self, data_path = r"F:\11_EFFORS\data\Edelsdorf.csv", out_path = "cross_indices"):
        self.data_path = data_path
        if os.path.isdir(out_path) == False:
            os.mkdir(out_path)
        self.out_path  = out_path
        # load data
        self.df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')

    def prepare_hincast_data(self, observations, hincastlen, forecastlen, start_index):
        X = []
        
        data = self.df[observations].values
        
        for i in range(start_index, len(data) - forecastlen):
            X.append(data[i-hincastlen:i])
            
        X = np.asarray(X)    
            
        return X

    def prepare_forecast_data(self, observations, training, hincastlen, forecastlen, start_index):
        X = []
        y = []
        
        data_X = self.df[observations].values
        data_y = self.df[training].values
        
        for i in range(start_index, len(data_X) - forecastlen):
            X.append(data_X[i:i+forecastlen])    
            y.append(data_y[i:i+forecastlen])

        X = np.asarray(X)
        y = np.asarray(y)
        
        return X, y

    #%%
    def create(self, n_sets = 7, hincast_lengths = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], forecast_len = 96, target_len = 96, oscilation_len=0):
        for hincast_len in hincast_lengths:
            print("prepareing index arrays")
            n_preCutOff  = hincast_len
            n_postCutOff = target_len
            n_samples = self.df.shape[0] - n_preCutOff
            
            i_splits = [n_preCutOff + int(x*n_samples/n_sets) for x in range(0,n_sets+1)]
            
            sets = {}
            for n_set, i in enumerate(range(1,n_sets+1)):
                i_set = np.arange(i_splits[i-1], i_splits[i] - n_postCutOff, dtype=int)
                i_hincast  = np.zeros((len(i_set), hincast_len, 1), dtype=int)
                i_forecast = np.zeros((len(i_set), forecast_len + oscilation_len, 1), dtype=int)
                i_target   = np.zeros((len(i_set), target_len   + oscilation_len, 1), dtype=int)
                for n,j in enumerate(i_set):
                    i_hincast[n,:,0]  = np.arange(j - hincast_len, 
                                                  j, 
                                                  dtype=int)
                    i_forecast[n,:,0] = np.arange(j - oscilation_len + 1, 
                                                  j + forecast_len + 1, 
                                                  dtype=int)
                    i_target[n,:,0]   = np.arange(j - oscilation_len + 1, 
                                                  j + target_len   + 1, 
                                                  dtype=int)
                
                if np.max(i_forecast) != i_splits[i] - 1:
                    print(f"index error: {np.max(i_forecast)} / {i_splits[i] - 1}")
                if np.min(i_hincast) < 0:
                    print("index error: index below 0")
            
                # sets[n_set] = (i_hincast.tolist(), i_forecast.tolist(), i_target.tolist())
                sets[n_set] = (i_hincast, i_forecast, i_target)
            
            print("set sizes (hincast, forecast, target):")
            for key in sets.keys():
                #print(f"    set {key}: ({np.array(sets[key][0]).shape}, {np.array(sets[key][1]).shape}, {np.array(sets[key][2]).shape})")
                print(f"    set {key}: ({sets[key][0].shape}, {sets[key][1].shape}, {sets[key][2].shape})")
                # hincast_i  = np.arange(idx_splits[n-1], idx_splits[n], dtype=int)
            
            
            dic = {"sets"  : sets,
                   "params": {"n_sets"       : n_sets,
                              "hincast_len"  : hincast_len,
                              "forecast_len" : forecast_len + oscilation_len,
                              "target_len"   : target_len + oscilation_len,
                }}
            with open(os.path.join(self.out_path, f'cross_indices_{hincast_len}.pckl'), 'wb') as fp:
                pickle.dump(dic, fp)
            #with open(os.path.join(self.out_path, f'cross_indices_{hincast_len}.json'), 'w') as fp:
            #    json.dump(dic, fp, sort_keys=True, indent=4)
            print('dictionary saved successfully to file')

