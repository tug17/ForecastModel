import os
num_threads = 16
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "16"

import tensorflow as tf
import keras_tuner
from tensorboard.plugins.hparams import api as tb_hp

from datetime import datetime

from ForecastModel.data.models import DataModelCV
from ForecastModel.models import CustomARIMA as architecture
from ForecastModel.tuners import MyARIMATuner
#

max_trials    = 100
inital_trials = 60
overwrite     = True
osc_offset    = 0

# paths 
TB_LOG_PATH = r"/home/sebastian/working_dir/Dissertation/LSTM_correction/ForecastModel/log"
CSV_PATH    = r"/home/sebastian/working_dir/Dissertation/LSTM_correction/data_preparation/train_data/Edelsdorf.csv"
CROSS_INDICES_PATH = r"/home/sebastian/working_dir/Dissertation/LSTM_correction/ForecastModel/indices_" + f"{osc_offset}"

CURRENT_TIME = datetime.strftime(datetime.now(), "%Y%m%d")
# CURRENT_TIME = "TESTING"
TB_LOG_PATH = os.path.join(TB_LOG_PATH, "fine_" + CURRENT_TIME + "_hindcast")

# define hp ranges
def call_model(hp):
    hyperparameter = {
           "p"    : hp.Int("p",      min_value=1, max_value=5, step=1),
           "d"    : hp.Int("d",      min_value=0, max_value=3, step=1),
           "q"    : hp.Int("q",      min_value=0, max_value=5, step=1),
           "forecast_len" : 96,
           "hindcast_length"    : hp.Int("hindcast_length", min_value=50, max_value=500, step=25)}
    
    model = architecture.build_model(hyperparameter)
    
    return model
# 
# create paths

if os.path.isdir(TB_LOG_PATH) == False:
    os.mkdir(TB_LOG_PATH)
    os.mkdir(os.path.join(TB_LOG_PATH, "logs"))
    os.mkdir(os.path.join(TB_LOG_PATH, "hp"))

# init datamodel 
dm = DataModelCV(CSV_PATH,
               target_name       = "qmeasval",
               hincast_features  = ['qsim','qmeastrain'],
               forecast_features = ['qsim'],
               )

# init hyperparameter object
hp = keras_tuner.HyperParameters()

# init tuner
tuner = MyARIMATuner(
    hypermodel         = call_model,
    objective          = "val_loss",
    max_trials         = max_trials,
    num_initial_points = inital_trials,
    overwrite          = overwrite,
    directory          = TB_LOG_PATH,
    # distribution_strategy = tf.distribute.MirroredStrategy(),
    project_name       = "hp",
    )

num_epochs = 100
patience = 30

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
