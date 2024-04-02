#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

#############################
#         Classes
#############################
class DataModelCV:
    def __init__(self, csv_path, target_name, hincast_features, forecast_features):
        self.csv_path = csv_path 
        self.target            = [target_name]
        self.hincast_features  = hincast_features
        self.forecast_features = forecast_features
        
        self.params = {"n_features_hc" : len(hincast_features),
                       "n_features_fc" : len(forecast_features),
            }
        
        self.df = None
        
    def loadCSV(self):
        self.df = pd.read_csv(self.csv_path, parse_dates=['time'], index_col='time')
        if self.df.index.tz == None:
            # make TZ aware
            print("datetimes set to UTC+0000")
            self.df.index = self.df.index.tz_localize("Europe/London", ambiguous='raise').tz_convert("UTC")
        else:
            self.df.index = self.df.index.tz_convert("UTC")
    
    def loadCrossIndices(self, filename='cross_indices.json'):
        # Read dictionary pkl file
        with open(filename, 'rb') as fp:
            dic = pickle.load(fp)
        #with open(filename, 'r') as fp:
        #    dic = json.load(fp)
        print('dictonary loaded')
        self.sets   = dic["sets"]
        #for key in self.sets.keys()
        #    self.sets[key] = (np.array(self.sets[key][0]), np.array(self.sets[key][1]), np.array(self.sets[key][2]))
        self.params.update(dic["params"])
        
    def getDataSet(self, n_set, scale=False, shuffle=False):
        if type(n_set) == type(list()):
            hi, fi, yi = [], [], []
            for n in n_set:
                hn, fn, yn = self.sets[n]
                hi.append(hn)
                fi.append(fn)
                yi.append(yn)
            hi = np.concatenate(hi, axis=0)
            fi = np.concatenate(fi, axis=0)
            yi = np.concatenate(yi, axis=0)
        else:
            hi, fi, yi = self.sets[n_set]
        
        sorting = np.arange(yi.shape[0])
        if shuffle:
            np.random.shuffle(sorting)

        Xh = self.getWithIndexArray(self.hincast_features, hi[sorting,:,:])
        Xf = self.getWithIndexArray(self.forecast_features, fi[sorting,:,:])
        y  = self.getWithIndexArray(self.target, yi[sorting,:,:])
        
        # print("dataset loaded")
        # print(Xh.shape)
        # print(Xf.shape)
        # print(y.shape)
        dataset = ((Xh, Xf), y)
        
        if scale:
            dataset = self.applyScaler(dataset, scale)
        return dataset
    
    def getTimeSet(self, n_set, depth=0):
        if type(n_set) == type(list()):
            hi, fi, yi = [], [], []
            for n in n_set:
                hn, fn, yn = self.sets[n]
                hi.append(hn)
                fi.append(fn)
                yi.append(yn)
            hi = np.concatenate(hi, axis=0)
            fi = np.concatenate(fi, axis=0)
            yi = np.concatenate(yi, axis=0)
        else:
            hi, fi, yi = self.sets[n_set]
            
        timeset = (self.df.index[hi[:,depth,0]], 
                   self.df.index[fi[:,depth,0]], 
                   self.df.index[yi[:,depth,0]],
                   )
        
        return timeset
    
    def getFeatureSet(self, n_set, feature_name, depth=0):
        if type(n_set) == type(list()):
            hi, fi, yi = [], [], []
            for n in n_set:
                hn, fn, yn = self.sets[n]
                hi.append(hn)
                fi.append(fn)
                yi.append(yn)
            hi = np.concatenate(hi, axis=0)
            fi = np.concatenate(fi, axis=0)
            yi = np.concatenate(yi, axis=0)
        else:
            hi, fi, yi = self.sets[n_set]
        
        featureset = (self.df[feature_name].iloc[hi[:,depth,0]], 
                   self.df[feature_name].iloc[fi[:,depth,0]], 
                   self.df[feature_name].iloc[yi[:,depth,0]],
                   )
        
        return featureset
    
    def getWithIndexArray(self, feat, idx):
        array = []
        for f in feat:
            array.append(self.df[f].values[idx])
            
        return np.concatenate(array, axis=2)
    
    def fitScaler(self, n_set):
        X, y = self.getDataSet(n_set)
        scaler_hincast  = MinMaxScaler()
        scaler_forecast = MinMaxScaler()
        self.scaler_hincast  = scaler_hincast.fit(X[0][:,0,:])
        self.scaler_forecast = scaler_forecast.fit(X[1][:,0,:])
        
    def applyScaler(self, dataset, hindcast_scale_index=True):  
        X, y = dataset
        Xh, Xf = X
        
        if type(hindcast_scale_index) == type([]):
            idx_feats = hindcast_scale_index
        else:
            idx_feats = [x for x in range(Xh.shape[1])]
        
        for n in idx_feats:
            Xh[:,n,:] = self.scaler_hincast.transform(Xh[:,n,:])
        for n in range(Xf.shape[1]):
            Xf[:,n,:] = self.scaler_forecast.transform(Xf[:,n,:])
        # print("scaling applied")
        return ((Xh, Xf), y)
    
    def getCrossValidSets(self, n_sets):
        cross_sets = {}
        for n in range(n_sets-2):
            cross_sets[n] = {
                 "train"      : [x for x in range(n+1)],
                 "valid"      : n+1,
                 "train_valid": [x for x in range(n+2)],
                 "test"       : n+2,
                 }
        return cross_sets
        
    def main(self, filename='cross_indices.pkl', verbose=1):
        self.loadCSV()
        self.loadCrossIndices(filename=filename)
        
        self.cross_sets = self.getCrossValidSets(self.params["n_sets"])
        
        # if verbose == 1:
        #     print("cross_validation sets:")
        #     print(self.cross_sets)
        
        self.fitScaler(0)
