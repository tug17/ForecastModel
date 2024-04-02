#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sebastian Gegenleithner
"""

#############################
#         Imports
#############################
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

from ForecastModel.utils.metrics import evaluate_multistep, calculate_bias, calculate_kge, calculate_nse, calculate_rmse

#############################
#         Functions
#############################
# split the data folds
def split_data(df):
    train_df = df[:-69806]
    val_df = df[-69806:-34903]
    test_df = df[-34903:]
    return train_df,val_df, test_df
    
# prepare hindacast data    
def prepare_hincast_data(df, observations,hincastlen,forecastlen, start_index):
    X = []
    data = df[observations].values
    for i in range(start_index, len(data) - forecastlen):
        X.append(data[i-hincastlen:i])
    X = np.asarray(X)    
    return X

# prepare forecast data
def prepare_forecast_data(df, observations,training,hincastlen,forecastlen, start_index): 
    X = []
    y = []
    data_X = df[observations].values
    data_y = df[training].values
    for i in range(start_index, len(data_X) - forecastlen):
        X.append(data_X[i:i+forecastlen])    
        y.append(data_y[i:i+forecastlen])
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

#############################
#         Classes
#############################
# custom ARIMA model with
# model parameters from optimization
class customARIMA:
    def __init__(self, 
                 p = 14,
                 d = 1, 
                 q = 13,
                 hindcast_data = ['qsim','qmeastrain'],
                 forecast_data = ['qmeastrain'],
                 hindcast_len = 200,
                 forecast_len = 96):
        self.p = p
        self.d = d
        self.q = q
        self.order = (self.p,self.d,self.q)
        self.hindcast_data = hindcast_data
        self.forecast_data = forecast_data
        self.hindcast_len = hindcast_len
        self.forecast_len = forecast_len
    # plot a single forecast with index
    def plot_fc(self,index):
        fig, axs = plt.subplots()
        axs.plot(np.arange(-self.hindcast_len + 1,1), self.X_test_hc[index][:,1], color = 'k', linestyle = '--', label = 'qmeas')
        axs.plot(np.arange(-self.hindcast_len + 1,1), self.X_test_hc[index][:,0], color = 'grey', linestyle = '--', label = 'qsim')
        axs.plot(np.arange(1,self.forecast_len+1),self.y_pred[index], 'b', label = 'Forecast')
        axs.plot(np.arange(1,self.forecast_len+1),self.y_test_fc[index], 'k')
        axs.plot(np.arange(1,self.forecast_len+1), self.X_test_fc[index][:,0], 'grey')
        axs.grid()
        axs.legend()        
    # check model
    def check_model(self):
        print(self.fitted_model.summary())
    # fit model
    def fit(self, train_df):
        X_train = train_df[self.hindcast_data]
        # compute residuals
        residual_hind = X_train['qsim'].values - X_train['qmeastrain'].values
        model = SARIMAX(residual_hind, 
                        order=self.order, 
                        enforce_stationarity = False, 
                        enforce_invertibility = False)
        self.fitted_model = model.fit(maxiter = 5000)
        print(self.fitted_model.summary())
    # make predictions
    def predict(self, df):
        # determine start index
        start_index = self.hindcast_len+1
        # prepare data
        self.X_test_hc = prepare_hincast_data(df, self.hindcast_data,self.hindcast_len,self.forecast_len, start_index)
        self.X_test_fc, self.y_test_fc = prepare_forecast_data(df, self.hindcast_data,self.forecast_data,self.hindcast_len,self.forecast_len, start_index)
        # init y predictions
        self.y_pred = np.empty((len(self.y_test_fc), self.forecast_len))
        # create empty list containing all forecasts
        all_forecasts = []
        # make predictions for all time steps
        for i in tqdm(range(0,len(self.y_test_fc))):
            # compute residuals
            error_i = self.X_test_hc[i][:,0] - self.X_test_hc[i][:,1]
            # apply model parameters from fitted arima model
            model_i = SARIMAX(error_i, order=self.order).filter(self.fitted_model.params)
            # make prediction
            forecast_error = model_i.get_forecast(steps=self.forecast_len).predicted_mean
            # correct hydrologic modeling results
            corrected_fc = self.X_test_fc[i][:,0] - forecast_error
            self.y_pred[i] = corrected_fc
            # add results datetime, corrected forecasts, measurements, and results of the original hydrologic model
            app_arr = [df.index[i + start_index]] + corrected_fc.tolist() + self.y_test_fc[i].reshape(96).tolist() + self.X_test_fc[i][:,0].reshape(96).tolist()
            all_forecasts.append(app_arr)
        # return the corrected forecasts, the observations, and the list of all forecasts
        return self.y_pred, self.y_test_fc, all_forecasts
    
#############################
#         Main
#############################
# path to training data
data_path = r'training.csv'
df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
# forecasts length in time steps
forecast_len = 96
# specifiy which folds we calculate
folds = [[2011,2012,2013],
         [2011,2012,2013,2014],
         [2011,2012,2013,2014,2015],
         [2011,2012,2013,2014,2015,2016],
         [2011,2012,2013,2014,2015,2016,2017]]

# loop over all folds
k = 1
for fold in folds:
    print('Starting fold %s' % k)
    # cut dataframe
    df_i = df[df.index.year.isin(fold)]
    # split data
    train_df, val_df, test_df = split_data(df_i)
    # create model and fit to training data
    model = customARIMA()
    model.fit(train_df)
    # predict for validation period
    y_pred_val, y_test_val, all_forecasts_val = model.predict(val_df)
    # predicti for testing period
    y_pred, y_test, all_forecasts = model.predict(test_df)
    # create output df
    all_fc_df = pd.DataFrame(all_forecasts, columns= ['Time'] + ['fc%s' % i for i in range(0,96)] + ['obs%s' % i for i in range(0,96)] + ['sim%s' % i for i in range(0,96)])
    all_fc_df.set_index('Time', inplace=True)
    # reshape y test
    y_test = y_test.reshape(len(y_pred), forecast_len)
    y_test_val = y_test_val.reshape(len(y_pred_val), forecast_len)
    # check physical bounds
    # below zero
    ind_below_zero = np.where(y_pred < 0)
    # number of forecast that contain a value below zero
    num_below_zero = len(np.unique(ind_below_zero[0]))
    # above extreme hight
    ind_above_high = np.where(y_pred > 100.)
    num_above_high = len(np.unique(ind_above_high[0]))
    # number violating physical constraints
    num_phys = num_below_zero + num_above_high
    print('%s forecasts violated the physical constraints' % num_phys)
    # check for invalid
    nan_indices = np.isnan(y_pred)
    nan_places = np.where(nan_indices)
    unique_nan_indices = np.unique(nan_places)
    num_nan = len(unique_nan_indices)
    print('%s forecasts produced nan' % num_nan)
    mask = np.ones(y_pred.shape[0], dtype=bool)
    mask[unique_nan_indices] = False
    # evaluate validation set
    losses_nse_val = []
    for i in range(0,forecast_len):
        nse_i = calculate_nse(y_test_val[:,i], y_pred_val[:,i])
        losses_nse_val.append(nse_i)
    losses_kge_val = []
    for i in range(0,forecast_len):
        kge_i = calculate_kge(y_test_val[:,i], y_pred_val[:,i])
        losses_kge_val.append(kge_i)
    losses_bias_val = []
    for i in range(0,forecast_len):
        bias_i = calculate_bias(y_test_val[:,i], y_pred_val[:,i])
        losses_bias_val.append(bias_i)    
    
    # evalute training set
    # evaluate only values that don't include nans
    losses_nse = []
    for i in range(0,forecast_len):
        nse_i = calculate_nse(y_test[:,i], y_pred[:,i])
        losses_nse.append(nse_i)
    losses_kge = []
    for i in range(0,forecast_len):
        kge_i = calculate_kge(y_test[:,i], y_pred[:,i])
        losses_kge.append(kge_i)
    losses_bias = []
    for i in range(0,forecast_len):
        bias_i = calculate_bias(y_test[:,i], y_pred[:,i])
        losses_bias.append(bias_i)

    print('NSE val: %s, NSE test: %s' % (np.mean(losses_nse_val),np.mean(losses_nse)))
    print('KGE val: %s, KGE test: %s' % (np.mean(losses_kge_val),np.mean(losses_kge)))
    print('BIAS val: %s, BIAS test: %s' % (np.mean(losses_bias_val),np.mean(losses_bias)))
    
    # pickle all forecasts
    all_fc_df.to_pickle('forecast_%s.pckl' % k)
    
    k+=1

# plot a single forecast   
# model.plot_fc(20848)

# plot forecasts
#arima_q0 = all_fc_df['fc%s' % 20].values
#measured_q0 = all_fc_df['obs%s' % 20].values
#hydro_q0 = all_fc_df['sim%s' % 20].values

#plt.plot(arima_q0, color = 'r')
#plt.plot(measured_q0, 'k-')
#plt.plot(hydro_q0, color = 'b')

#nse_testing = calculate_nse(measured_q0, arima_q0)

