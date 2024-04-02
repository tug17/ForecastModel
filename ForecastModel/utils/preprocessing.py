#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import pickle
# import json

import os

#############################
#         Classes
#############################
class CreateIndices:
    def __init__(self, data_path = r"F:\11_EFFORS\data\Edelsdorf.csv", out_path = "cross_indices"):
        self.data_path = data_path
        if os.path.isdir(out_path) == False:
            os.mkdir(out_path)
        self.out_path  = out_path
        # load data
        self.df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
        self.df = self.df.set_index(pd.to_datetime(self.df.index))
        self.df["n_idx"] = np.arange(self.df.shape[0], dtype=int)
        try: 
            self.mask = self.df["is_peak_flow"].values.tolist()
        except:
            print("no masking column found")
            self.mask = [True for x in range(self.df.shape[0])]

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
    def create(self, n_sets = 7, hincast_lengths = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], forecast_len = 96, target_len = 96, oscilation_len=0, dtime_secs=15*60):
        for hincast_len in hincast_lengths:
            print("prepareing index arrays")
            
            hindcast_delta  = hincast_len * timedelta(seconds=dtime_secs)
            forecast_delta  = forecast_len* timedelta(seconds=dtime_secs)
            target_delta    = target_len  * timedelta(seconds=dtime_secs)
            
            start_date = self.df.index[0]  + hindcast_delta
            end_date   = self.df.index[-1] # - np.max((forecast_delta, target_delta))
            
            
            i_splits = [(start_date + x * (end_date-start_date) / n_sets).round('1d') for x in range(0,n_sets+1)]
            
            df_part = self.df.loc[self.mask]
            
            sets = {}
            for n_set, i in enumerate(range(1,n_sets+1)):
                print(f"processing set {n_set+1}")
                set_mask = (df_part.index > i_splits[i-1]) & (df_part.index < i_splits[i] - forecast_delta)
                df_part_set = df_part.loc[set_mask]
                
                #set_mask = (self.df.index > i_splits[i-1]) & (self.df.index <= i_splits[i] - forecast_delta)
                #df_set = self.df.loc[set_mask]
                
                n_samples = df_part_set.shape[0]
                
                i_hincast  = np.zeros((n_samples, hincast_len, 1), dtype=int)
                i_forecast = np.zeros((n_samples, forecast_len + oscilation_len, 1), dtype=int)
                i_target   = np.zeros((n_samples, target_len   + oscilation_len, 1), dtype=int)
                
                for n,date in enumerate(df_part_set.index):
                    hindcast_dates = pd.date_range(date - hindcast_delta, date, freq=f"{dtime_secs:d}s")[1:]
                    forecast_dates = pd.date_range(date, date + forecast_delta, freq=f"{dtime_secs:d}s")[1:]
                    target_dates   = pd.date_range(date, date + target_delta,   freq=f"{dtime_secs:d}s")[1:]
                    
                    if forecast_dates[-1] > i_splits[i]:
                        print("set limit reached")
                        np.delete(i_hincast,  n, axis=0)
                        np.delete(i_forecast, n, axis=0)
                        np.delete(i_target,   n, axis=0)
                        continue
                    
                    try:
                        i_hincast[n,:,0]  = self.df.loc[hindcast_dates, "n_idx"].values
                        i_forecast[n,:,0] = self.df.loc[forecast_dates, "n_idx"].values
                        i_target[n,:,0]   = self.df.loc[target_dates,   "n_idx"].values
                    except:
                        print(f"an error occured in sample {n}: sample was removed")
                        np.delete(i_hincast,  n, axis=0)
                        np.delete(i_forecast, n, axis=0)
                        np.delete(i_target,   n, axis=0)
                    
                
                if np.max(i_forecast) >= self.df["n_idx"].values[-1]:
                    print(f"index error: {np.max(i_forecast)}")
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

