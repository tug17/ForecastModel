#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
import os
import numpy as np
import json
import pandas as pd

#############################
#         Functions
#############################
def load_metrics(path):
    with open(path, "r") as f:
        metrics = json.load(f)
    return metrics

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

def regroup_metrics(directory=r"F:\11_EFFORS\python\models"):
    for path in os.listdir(directory):
        if path == "old":
            continue
        metric = {"eval": {},
                 "test": {}}
        for key in ["nse", "kge", "bias"]:
            metric["test"][key] = np.loadtxt(os.path.join(directory, path, f"metric_{key}.txt"), delimiter=",").tolist()

        with open(os.path.join(directory, path, "metrics.txt"), "w+") as f:
            json.dump(metric, f)

def df2latex(df, file):
    print(file)
    with open(file, "w") as f:
        f.write("index&"+"&".join(df.columns)+r"\\")
        f.write("\n")
        for n, row in df.iterrows():
            f.write(str(n))
            for num in row.values:
                if np.abs(num) >= 100:
                    f.write(f"&{int(num): 6d}")
                elif np.abs(num) >= 10:
                    f.write(f"&{num: 6.1f}")
                elif np.abs(num) >= 1:
                    f.write(f"&{num: 6.2f}")
                else:
                    f.write(f"&{num: 6.3f}")
            f.write(r"\\")
            f.write("\n")

#############################
#         Classes
#############################
# class to handle plotting easier
class Model:
    def __init__(self, name, model_folder, n_trial=-1, target_name="", feat_hindcast=[], feat_forecast=[], is_external_model= False, is_final_model= False, color="r", ls="-"):
        self.name  = name
        self.color = color
        self.ls    = ls
        self.is_external_model = is_external_model
        
        if is_final_model:
            self.lg_path = model_folder
            self.hp_path = model_folder
            if is_external_model:
                is_external_model
                # do nothing
            else:
                with open(os.path.join(self.lg_path, "features.txt"), "r") as f:
                    dic = json.load(f)
                self.target_name   = dic["target_name"]
                self.feat_hindcast = dic["feat_hindcast"]
                self.feat_forecast = dic["feat_forecast"]
        else:
            self.lg_path = os.path.join(model_folder, "log", f"trial_{n_trial:02d}")
            self.hp_path = os.path.join(model_folder,  "hp", f"trial_{n_trial:02d}")
            self.target_name   = target_name
            self.feat_hindcast = feat_hindcast
            self.feat_forecast = feat_forecast