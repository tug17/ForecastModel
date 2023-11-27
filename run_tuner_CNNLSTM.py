import os
num_threads = 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

import tensorflow as tf
import keras_tuner
from tensorboard.plugins.hparams import api as tb_hp

from datetime import datetime

from ForecastModel.data.models import DataModelCV
from ForecastModel.models import Hindcast as architecture
from ForecastModel.tuners import MyTuner
#%%
# training settings
num_epochs = 100
patience   = 10
osc_offset = 12 #0 #24

max_trials    = 100
inital_trials = 60
overwrite     = True

# paths 
TB_LOG_PATH = r"F:\11_EFFORS\python\tb"
CSV_PATH    = r"F:\11_EFFORS\data\Edelsdorf.csv"
CROSS_INDICES_PATH = r"F:\11_EFFORS\data\indices_" + f"{osc_offset}"

CURRENT_TIME = datetime.strftime(datetime.now(), "%Y%m%d")
# CURRENT_TIME = "TESTING"
TB_LOG_PATH = os.path.join(TB_LOG_PATH, "fine_" + CURRENT_TIME + "_cnnlstm")


#%% define hp ranges
def call_model(hp):
    # hyperparameter = {
          # "cnn_filter"    : hp.Int("cnn_filter",      min_value=4, max_value=24, step=2),
          # "cnn_kernel"    : hp.Int("cnn_kernel",      min_value=1, max_value=9, step=2),
          # "cnn_stride"    : 1,                          #hp.Int("cnn_stride",      min_value=1, max_value=9, step=2),
          # "pool_size"     : hp.Int("pool_size",       min_value=1, max_value=9, step=2),
          # "pool_stride"   : 1,                          #hp.Int("pool_stride",     min_value=1, max_value=9, step=2),
          # "dropout_rate"  : 0.05,                       #hp.Float("dropout_rate",  min_value=0.0, max_value=0.3, step=0.025),
          # "lstm_unit"     : hp.Int("lstm_unit",       min_value=5, max_value=50, step=5),
          # "lstm_dropout"  : 0.05,                       #hp.Float("lstm_dropout",  min_value=0.0, max_value=0.3, step=0.025),
          # "lr"            : hp.Float("lr",            min_value=0.0001, max_value=0.0017, step=0.0001),
          # "batch_size"    : hp.Int("batch_size",      min_value=200, max_value=400, step=50),
          # "retrain_epochs": hp.Fixed("retrain_epochs",value=10),
          # "osc_length"    : hp.Fixed("osc_length",    value=0),
          # "hindcast_len"  : hp.Int("hindcast_length", min_value=24, max_value=120, step=24),
          # "forecast_len"  : 96+osc_offset,
          # "target_len"    : 96+osc_offset,
          # "n_features_hc" : 5, 
          # "n_features_fc" : 4, 
          # }
    hyperparameter = {
          "cnn_filter"    : hp.Fixed("cnn_filter",      value=4),
          "cnn_kernel"    : hp.Fixed("cnn_kernel",      value=7),
          "cnn_stride"    : 1,                          #hp.Int("cnn_stride",      min_value=1, max_value=9, step=2),
          "pool_size"     : hp.Fixed("pool_size",       value=9),
          "pool_stride"   : 1,                          #hp.Int("pool_stride",     min_value=1, max_value=9, step=2),
          "dropout_rate"  : hp.Float("dropout_rate",    min_value=0.04, max_value=0.06, step=0.01),
          "lstm_unit"     : hp.Fixed("lstm_unit",       value=5),
          "lstm_dropout"  : hp.Float("lstm_dropout",    min_value=0.04, max_value=0.06, step=0.01),
          "lr"            : hp.Float("lr",              min_value=0.000165, max_value=0.00175, step=0.00025),
          "batch_size"    : hp.Fixed("batch_size",      value=200),
          "retrain_epochs": hp.Fixed("retrain_epochs",  value=10),
          "osc_length"    : hp.Fixed("osc_length",      value=0),
          "hindcast_len"  : hp.Fixed("hindcast_length", value=72),
          "forecast_len"  : 96+osc_offset,
          "target_len"    : 96+osc_offset,
          "n_features_hc" : 5, 
          "n_features_fc" : 4, 
          }
          
    # hyperparameter["cnn_stride"]  = int(hyperparameter["cnn_kernel"]/2) + 1
    # hyperparameter["pool_stride"] = int(hyperparameter["pool_size"]/2) + 1
    
    model = architecture.build_model(hyperparameter)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameter["lr"])
    model.compile(optimizer=optimizer, 
               loss='mean_squared_error')
    
    return model

    
    
#%% 
# create paths

if os.path.isdir(TB_LOG_PATH) == False:
    os.mkdir(TB_LOG_PATH)
    os.mkdir(os.path.join(TB_LOG_PATH, "logs"))
    os.mkdir(os.path.join(TB_LOG_PATH, "hp"))

# init datamodel 
dm = DataModelCV(r"F:\11_EFFORS\data\Edelsdorf.csv",
               target_name       = "qmeasval",
               hincast_features  = ['qsim','pmax','tmean','pmean','qmeastrain'],
               forecast_features = ['qsim','pmax','tmean','pmean'],
               )

# init hyperparameter object
hp = keras_tuner.HyperParameters()

# init tuner
tuner = MyTuner(
    hypermodel         = call_model,
    objective          = "val_loss",
    max_trials         = max_trials,
    num_initial_points = inital_trials,
    overwrite          = overwrite,
    directory          = TB_LOG_PATH,
    # distribution_strategy = tf.distribute.MirroredStrategy(),
    project_name       = "hp",
    )

# start tuner
tuner.search(data_model = dm,
          epochs  = num_epochs, 
          verbose = 1,
          callbacks=[
                     # tf.keras.callbacks.EarlyStopping("loss", patience=3, min_delta=0.0002),
                     tf.keras.callbacks.EarlyStopping("val_loss", 
                                                      patience=patience, 
                                                      restore_best_weights=True)],
          cross_indices_path   = CROSS_INDICES_PATH, 
          tb_log_path          = TB_LOG_PATH,
          plot_fold_rst        = True, 
          save_fold_models     = True, 
          save_fold_prediction = False,
          )
