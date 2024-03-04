#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import os
import sys
# from statsmodels.tsa.arima.model import ARIMA

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(17)

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stderr = old_stderr

#%% 
class CNNLSTM:
    def __init__(self, random_seed=17):
        tf.keras.utils.set_random_seed(random_seed)
        
    def build_model(hp):
        ## input for hincast and forecast model
        ### input for hindcast is ignored !!!!
        inp_hincast  = tf.keras.layers.Input(shape=(hp["hindcast_len"], hp["n_features_hc"]))
        inp_forecast = tf.keras.layers.Input(shape=(hp["forecast_len"], hp["n_features_fc"]))
        
        ## forecast model
        # add the CNN layer
        hidden = tf.keras.layers.Conv1D(filters     = hp["cnn_filter"], 
                                        kernel_size = hp["cnn_kernel"], 
                                        strides     = hp["cnn_stride"],
                                        padding     = "valid",
                                        activation  = 'relu',
                                        )(inp_forecast)
        # add the maxpooling layer
        hidden = tf.keras.layers.MaxPooling1D(pool_size = hp["pool_size"], 
                                              strides   = hp["pool_stride"],
                                              padding   = "valid",
                                              )(hidden)
        # add dropout layer
        hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # add lstm and set inital state
        hidden = tf.keras.layers.LSTM(units  = hp["lstm_unit_1"],
                                             activation   ='tanh', 
                                             dropout      = hp["lstm_dropout_1"],
                                             return_sequences=True,
                                            )(hidden)
        
        if hp["lstm_unit_2"] != 0:
            hidden = tf.keras.layers.LSTM(units  = hp["lstm_unit_2"],
                                             activation   ='tanh', 
                                             dropout      = hp["lstm_dropout_2"],
                                             return_sequences=True,
                                            )(hidden)
        # flatten layer
        hidden = tf.keras.layers.Flatten()(hidden)
        
        # fully connected
        hidden = tf.keras.layers.Dense(hp["target_len"])(hidden)
        
        ## create model                               
        model = tf.keras.Model(inputs=[inp_hincast, inp_forecast], outputs=hidden)

        # model.summary()
        return model

#%% 
class Hindcast:
    def __init__(self, random_seed=17):
        tf.keras.utils.set_random_seed(random_seed)

    def build_model(hp):
        ## input for hincast and forecast model
        inp_hincast  = tf.keras.layers.Input(shape=(hp["hindcast_len"], hp["n_features_hc"]))
        inp_forecast = tf.keras.layers.Input(shape=(hp["forecast_len"], hp["n_features_fc"]))
        
        ## hincast model
        hidden  = inp_hincast
        
        # add LSTM layer
        lstm_hincast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                      activation   = 'tanh', 
                                      dropout      = 0, #hp["lstm_dropout"],
                                      return_state = True,
                                      # return_sequences = True,
                                      )(hidden)

        ## Pass the last hidden state and cell state to another LSTM layer
        _, last_hidden_state, last_cell_state = lstm_hincast
        
        last_hidden_state = tf.keras.layers.Dense(hp["lstm_unit"], activation ='linear')(last_hidden_state)
        last_cell_state   = tf.keras.layers.Dense(hp["lstm_unit"], activation ='linear')(last_cell_state)
        
        ## forecast model
        hidden = inp_forecast
        
        # add lstm and set inital state
        lstm_forecast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                             activation   = 'tanh', 
                                             dropout      = 0, #hp["lstm_dropout"],
                                             return_sequences=True,
                                             #go_backwards=True,
                                            )(hidden,
                                              initial_state=[last_hidden_state, last_cell_state],
                                             )
        # flatten layer
        hidden = tf.keras.layers.Flatten()(lstm_forecast)
        
        hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # fully connected
        hidden = tf.keras.layers.Dense(hp["target_len"], activation='relu')(hidden)
        
        ## create model                               
        model = tf.keras.Model(inputs=[inp_hincast, inp_forecast], outputs=hidden)

        #model.summary()
        return model


class HindcastCNN:
    def __init__(self, random_seed=17):
        tf.keras.utils.set_random_seed(random_seed)

    def build_model(hp):
        ## input for hincast and forecast model
        inp_hincast  = tf.keras.layers.Input(shape=(hp["hindcast_len"], hp["n_features_hc"]))
        inp_forecast = tf.keras.layers.Input(shape=(hp["forecast_len"], hp["n_features_fc"]))
        
        ## hincast model
        # add the CNN layer
        hidden = tf.keras.layers.Conv1D(filters     = hp["cnn_filter"], 
                                         kernel_size = hp["cnn_kernel"], 
                                         strides     = hp["cnn_stride"],
                                         padding     = "valid",
                                         activation='tanh')(inp_hincast)
        
        # # add max pooling
        hidden = tf.keras.layers.MaxPooling1D(pool_size = hp["pool_size"], 
                                               strides   = hp["pool_stride"],
                                               padding     = "valid",
                                               )(hidden)
        
        # # add dropout
        # hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        #hidden  = inp_hincast
        # add LSTM layer
        lstm_hincast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                      activation   = 'tanh', 
                                      dropout      = 0, #hp["lstm_dropout"],
                                      return_state = True,
                                      # return_sequences = True,
                                      )(hidden)

        ## Pass the last hidden state and cell state to another LSTM layer
        _, last_hidden_state, last_cell_state = lstm_hincast
        
        last_hidden_state = tf.keras.layers.Dense(hp["lstm_unit"], activation ='linear')(last_hidden_state)
        last_cell_state   = tf.keras.layers.Dense(hp["lstm_unit"], activation ='linear')(last_cell_state)
        
        ## forecast model
        # add the CNN layer
        hidden = tf.keras.layers.Conv1D(filters     = hp["cnn_filter"], 
                                         kernel_size = hp["cnn_kernel"], 
                                         strides     = hp["cnn_stride"],
                                         padding     = "valid",
                                         activation='tanh',
                                         )(inp_forecast)
        # # add the maxpooling layer
        hidden = tf.keras.layers.MaxPooling1D(pool_size = hp["pool_size"], 
                                               strides   = hp["pool_stride"],
                                               padding     = "valid",
                                               )(hidden)
        # add dropout layer
        # hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # add lstm and set inital state
        #hidden = inp_forecast
        lstm_forecast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                             activation   = 'tanh', 
                                             dropout      = 0, #hp["lstm_dropout"],
                                             return_sequences=True,
                                             #go_backwards=True,
                                            )(hidden,
                                              initial_state=[last_hidden_state, last_cell_state],
                                             )
        # flatten layer
        hidden = tf.keras.layers.Flatten()(lstm_forecast)
        
        hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # fully connected
        hidden = tf.keras.layers.Dense(hp["target_len"], activation='relu')(hidden)
        # hidden = tf.keras.layers.Dense(hp["target_len"], activation='tanh')(hidden)
        
        ## create model                               
        model = tf.keras.Model(inputs=[inp_hincast, inp_forecast], outputs=hidden)

        #model.summary()
        return model

#%%
# Generic ARIMA class
class GenModel:
    # Initialisation
    def __init__(self,hp):
        # init hyperparameters
        self.p = hp["p"]
        self.d = hp["d"]
        self.q = hp["q"]
        self.forecast_len = hp["forecast_len"]
        self.seq_length = hp["hindcast_length"]
        # set order of the model p=AR, d=diff, q=MA
        self.order = (self.p,self.d,self.q)
    # Some dummy fitting function. Actually not required though
    def fit(self,X_train,y_train, **kwargs):
        # nothing to fit.
        print("No fitting required")
    # Perform predictions    
    def predict(self,X_test):
        # init y_pred tensor. We multiply by the maximum of the training data
        # this is done since the ARIMA model does not always converge. If it doesnt
        # the predictions are not replaced, thus containing the maximum of the observations
        # which results in unfavorable loss functions. Thus the optimizer should discard instances
        # for which no converge is frequenty observed automatically
        max_train = np.amax(X_test[1])
        y_pred = np.ones((len(X_test[0]), self.forecast_len)) * max_train
        # loop over all test data
        for i in tqdm(range(0,len(X_test[0]))):
            # compute the error i.e. simulation minus observations
            error_i = X_test[0][i][:,0] - X_test[0][i][:,1]
            # catch errors in ARIMA
            try:
                # supress convergence warnings of the ARIMA model
                with suppress_stderr():
                    model = ARIMA(error_i, order = self.order)
                    results = model.fit()
                    forecast_error = results.get_forecast(steps=self.forecast_len).predicted_mean
                    # forecast the error function
                    forecast_error = forecast_error.reshape(len(forecast_error))
                # simulations of the forecast
                sim_fc = X_test[1][i].reshape(len(forecast_error))
                # correct the simulations with the forecast of the error
                corrected_fc = sim_fc - forecast_error
                # get rid of nan values use very high values as a penalty instead
                corrected_fc = np.nan_to_num(corrected_fc, nan=max_train)
                # add forecast to y
                y_pred[i] = corrected_fc
            except:
                continue
        # return the results
        return y_pred

# This is just a wrapper class to fit to the structure of the tuner        
class CustomARIMA:
    # Initialisation, actually not required
    def __init__(self):
        pass
    # build the model for optimization
    def build_model(hp):
        return GenModel(hp)

