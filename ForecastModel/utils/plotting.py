#!/usr/bin/env python3
import tensorflow as tf

import os

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FuncFormatter

import numpy as np

def plot_model_architecture(model_path: str, figname="fig"):
model = tf.keras.models.load_model(model_path)

tf.keras.utils.plot_model(model,
                    to_file=os.path.join(PLOT_PATH, figname + '.png'),
                    show_shapes=True,
                    show_dtype=False,
                    show_layer_names=True,
                    rankdir="TB",
                    expand_nested=False,
                    dpi=200,
                    layer_range=None,
                    show_layer_activations=True,
                )
    return 0

    
def plot_metric_per_fold(metric_paths: list, figname="fig"):

    metrics = ["kge", "nse", "bias"]
    metric_labels = ["KGE", "NSE", "PBIAS"]
    ylims = [(0,1), (-2,1), (-20,10)]

    colors = ["r", "b", "k"]
    labels = ["LSTM", "Hindcast-LSTM", "ARIMA"]

    xx = np.arange(0,96) * 0.25

    fig, axes = plt.subplots(5,3,figsize=(8,10))

    for n, path in enumerate(paths):
        for j, metric in enumerate(metrics):
            metric  = np.loadtxt(os.path.join(path, f"metric_{metric}.txt"), delimiter=",")
            for i in range(5):
                axes[i,j].plot(xx, metric[i,:], color=colors[n], label=labels[n])
                
    for i in range(5):
        for j in range(3):
            if i == 0:
                axes[i,j].set_title(metric_labels[j])
            elif i == 4:
                axes[i,j].set_xlabel("lead time (h)")
            if j == 0:
                axes[i,j].set_ylabel(f"{2011+i:d} (fold {i+1})")
            if (j == 2) & (i == 2):
                axes[i,j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            
            axes[i,j].xaxis.set_major_locator(FixedLocator(np.arange(0,25,8)))
            if (j != 2):
                axes[i,j].yaxis.set_major_locator(FixedLocator(np.arange(0.5,1.1,0.1)))
            if (j == 1) & (i == 4):
                axes[i,j].yaxis.set_major_locator(FixedLocator(np.arange(-2,1.1,0.5)))
            
            axes[i,j].set_xlim((0,24))
            axes[i,j].grid("both")

    plt.draw()
    plt.show()
    
    fig.savefig(os.path.join(PLOT_PATH, figname + '.png'), 
            bbox_inches='tight')
            
    return 0