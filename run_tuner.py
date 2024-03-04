import os
# use async allocator to avoid OOM on GPU
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

import tensorflow as tf
import keras_tuner

from datetime import datetime

import json

from ForecastModel.data.models import DataModelCV
from ForecastModel.utils.losses import loss_nkge_nnse
from ForecastModel.models import Hindcast as architecture
from ForecastModel.tuners import MyTuner

tf.config.run_functions_eagerly(False)

# training settings
num_epochs      = 10
patience        = 3
max_trials      = 1
inital_trials   = 1
overwrite       = True

# paths
TB_LOG_PATH = r"tb"
DATA_PATH   = r"data\dummy_data.csv"
CROSS_INDICES_PATH = r"data\indices"

CURRENT_TIME = datetime.strftime(datetime.now(), "%Y%m%d")
TB_LOG_PATH = os.path.join(TB_LOG_PATH, CURRENT_TIME + "_random")

# definitions
features = {
    "target_name": 'qmeasval',
    "feat_hindcast": [
        'qsim',
        'pmax',
        'tmean',
        'pmean', 
        'qmeasval',
        ],
    "feat_forecast": [
        'qsim',
        'pmax',
        'tmean',
        'pmean',
        ],
    }

# create paths
if os.path.isdir(TB_LOG_PATH) == False:
    os.mkdir(TB_LOG_PATH)
    os.mkdir(os.path.join(TB_LOG_PATH, "logs"))
    os.mkdir(os.path.join(TB_LOG_PATH, "hp"))
    
# save feature list
print(os.path.join(TB_LOG_PATH, "features.txt"))
with open(os.path.join(TB_LOG_PATH, "features.txt"), "w") as f:
    json.dump(features, f)

# define hp search space
def call_model(hp):
    hyperparameter = {
           "dropout_rate"   : hp.Float("dropout_rate",  min_value=0.01, max_value=0.5),
           "lstm_unit"      : hp.Int("lstm_unit",       min_value=12,   max_value=32),
           "lstm_dropout"   : 0, #hp.Float("lstm_dropout",  min_value=0.0, max_value=0.3, step=0.025), removed due to cuDNN constraints
           "lr"             : hp.Float("lr", min_value=0.001, max_value=0.01, sampling="log"),
           "batch_size"     : hp.Int("batch_size",      min_value=200,max_value=1000, step=100),
           "retrain_epochs" : hp.Int("retrain_epochs",  min_value=0,  max_value=3,    step=1),
           "hindcast_len"   : hp.Int("hindcast_length", min_value=96, max_value=96*2, step=96),
           "forecast_len"   : 96,
           "target_len"     : 96,
           "n_features_hc"  : len(features["feat_hindcast"]), 
           "n_features_fc"  : len(features["feat_forecast"]), 
           }
    
    model = architecture.build_model(hyperparameter)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameter["lr"], 
                                         clipvalue=0.5, 
                                         clipnorm=0.001)
    model.compile(optimizer=optimizer, 
               #loss='mean_squared_error',
               loss = loss_nkge_nnse
               )
    
    return model

# init datamodel 
dm = DataModelCV(DATA_PATH,
               target_name       = features["target_name"],
               hincast_features  = features["feat_hindcast"],
               forecast_features = features["feat_forecast"],
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
    project_name       = "hp",
    )

# define lr scheudler
def scheduler(epoch, lr):
    if epoch == 0:
        return lr
    elif epoch < 4:
        return lr
    elif epoch < 20:
        return lr * 0.9
    else:
        return lr

# start tuner
tuner.search(
          data_model = dm,
          epochs  = num_epochs, 
          shuffle = True,
          verbose = 1,
          callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler),
                     tf.keras.callbacks.EarlyStopping("val_loss", 
                                                        patience=patience, # min_delta=0.001, 
                                                        restore_best_weights=True),
                                             ],
          cross_indices_path   = CROSS_INDICES_PATH, 
          tb_log_path          = TB_LOG_PATH,
          plot_fold_rst        = True, 
          save_fold_models     = True, 
          save_fold_prediction = False,
          )