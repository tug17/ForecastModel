#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(17)

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
        # add the CNN layer
        hidden = tf.keras.layers.Conv1D(filters     = hp["cnn_filter"], 
                                        kernel_size = hp["cnn_kernel"], 
                                        strides     = hp["cnn_stride"],
                                        padding     = "valid",
                                        activation='relu')(inp_hincast)
        
        # add max pooling
        hidden = tf.keras.layers.MaxPooling1D(pool_size = hp["pool_size"], 
                                              strides   = hp["pool_stride"],
                                              padding     = "valid",
                                              )(hidden)
        
        # add dropout
        hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # add LSTM layer
        lstm_hincast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                      activation   ='tanh', 
                                      dropout      = hp["lstm_dropout"],
                                      return_state = True,
                                      # return_sequences = True,
                                      )(hidden)
        
        ## Pass the last hidden state and cell state to another LSTM layer
        _, last_hidden_state, last_cell_state = lstm_hincast
        
        last_hidden_state = tf.keras.layers.Dense(hp["lstm_unit"])(last_hidden_state)
        last_cell_state   = tf.keras.layers.Dense(hp["lstm_unit"])(last_cell_state)
        
        ## forecast model
        # add the CNN layer
        hidden = tf.keras.layers.Conv1D(filters     = hp["cnn_filter"], 
                                        kernel_size = hp["cnn_kernel"], 
                                        strides     = hp["cnn_stride"],
                                        padding     = "valid",
                                        activation='relu',
                                        )(inp_forecast)
        # add the maxpooling layer
        hidden = tf.keras.layers.MaxPooling1D(pool_size = hp["pool_size"], 
                                              strides   = hp["pool_stride"],
                                              padding     = "valid",
                                              )(hidden)
        # add dropout layer
        hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # add lstm and set inital state
        lstm_forecast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                             activation   ='tanh', 
                                             dropout      = hp["lstm_dropout"],
                                             return_sequences=True,
                                            )(hidden,
                                              initial_state=[last_hidden_state, last_cell_state],
                                             )
                                              
        # flatten layer
        hidden = tf.keras.layers.Flatten()(lstm_forecast)
        
        # fully connected
        hidden = tf.keras.layers.Dense(hp["target_len"])(hidden)
        
        ## create model                               
        model = tf.keras.Model(inputs=[inp_hincast, inp_forecast], outputs=hidden)

        # model.summary()
        return model

#%% 
class ARIMA:
    def __init__(self, random_seed=17):
        tf.keras.utils.set_random_seed(random_seed)

