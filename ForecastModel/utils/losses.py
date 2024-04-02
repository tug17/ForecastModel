#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
import tensorflow as tf

#############################
#         Functions
#############################
def loss_nkge_nnse(y, y_hat):
    ## combinded NNSE and NKGE loss
    # normalized Nash–Sutcliffe efficiency
    # mean of observation
    mean_obs = tf.reduce_mean(y)

    # value importance
    #f = (y / 3)


    # Calculate numerator and denominator of NSE formula
    a = tf.reduce_sum(tf.square(y - y_hat))
    b = tf.reduce_sum(tf.square(y - mean_obs))

    # calculate NSE
    nse = 1 - a / b
    
    nnse = (1 / (2 - nse))

    ## modified and normalized Kling-Gupta Efficiency
    # means and ratios
    mean_hat = tf.reduce_mean(y_hat)
    mean_ratio = mean_hat / mean_obs
    
    # weighted std ratio
    a = tf.math.reduce_std(y_hat) / mean_hat
    b = tf.math.reduce_std(y)     / mean_obs
    weighted_std_ratio = a / b
    
    # correlation coefficient
    a = tf.reduce_sum((y - mean_obs) * (y_hat - mean_hat))
    b = tf.sqrt(tf.reduce_sum((y - mean_obs) ** 2)) * tf.sqrt(tf.reduce_sum((y_hat - mean_hat) ** 2))
    r = a / b

    # calculate KGE
    kge = 1.0 - tf.sqrt((r - 1.0)**2 + (weighted_std_ratio - 1.0)**2 + (mean_ratio - 1.0)**2)
    nkge = (1 / (2 - kge))
    
    
    return (2 - nkge - nnse)

def loss_nkge(y, y_hat):
    ## modified and normnalized Kling-Gupta Efficiency
    # means and ratios
    mean_obs = tf.reduce_mean(y) 
    mean_hat = tf.reduce_mean(y_hat)
    mean_ratio = mean_hat / mean_obs
    
    # weighted std ratio
    a = tf.math.reduce_std(y_hat)
    b = tf.math.reduce_std(y)
    std_ratio = a / b
    
    # correlation coefficient
    a = tf.reduce_sum((y - mean_obs) * (y_hat - mean_hat))
    b = tf.sqrt(tf.reduce_sum((y - mean_obs) ** 2)) * tf.sqrt(tf.reduce_sum((y_hat - mean_hat) ** 2))
    r = a / b

    # calculate KGE
    kge = 1.0 - tf.sqrt((r - 1.0)**2 + (2*(std_ratio - 1.0))**2 + (mean_ratio - 1.0)**2)
    nkge = (1 / (2 - kge))
    
    
    return 1 - nkge

def loss_quantile_98(y, y_hat):
    ## loss funtion to minimize 98% quantile
    q = 0.75
    e = y - y_hat
    return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))