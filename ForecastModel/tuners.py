#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
import os
import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import keras_tuner

from tensorboard.plugins.hparams import api as tb_hp

from datetime import datetime
import matplotlib.pyplot as plt

import json

from ForecastModel.utils.metrics import evaluate_multistep, calculate_bias, calculate_kge, calculate_nse, calculate_rms, calculate_kge5alpha

#############################
#         Classes
#############################
#%
class MyTuner(keras_tuner.BayesianOptimization):
    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        with tf.summary.create_file_writer(os.path.join(self.tb_log_path, r"logs\trial_"+f"{trial.trial_id}")).as_default():
            score = trial.score
            hparams = trial.hyperparameters.get_config()['values']
            tf.summary.scalar('score', score, step=1)
            tb_hp.hparams(hparams)
        
    def save_model(self, trial, model):
        model.save(os.path.join(self.tb_log_path, "hp", f"trial_{trial.trial_id}", "model.keras"))
        
    def save_model_fold(self, trial, fold_id, model):
        model.save(os.path.join(self.tb_log_path, "hp", f"trial_{trial.trial_id}", f"model_fold_{fold_id}.keras"))
        
    def run_trial(self, trial, data_model, verbose, epochs, callbacks, cross_indices_path, tb_log_path, shuffle=False, plot_fold_rst=True, save_fold_models=False, save_fold_prediction=True):
        print(trial.trial_id)
        
        # set tb_log_path
        self.tb_log_path = tb_log_path
        
        current_log_path = os.path.join(tb_log_path, r"logs\trial_"+f"{trial.trial_id}")
        
        # load hyperparameters
        hp = trial.hyperparameters
        
        # get data model
        print(hp)
        hindcast_length = hp["hindcast_length"]
        data_model.main(os.path.join(cross_indices_path, f"cross_indices_{hindcast_length}.pckl"), verbose)
        
        metric_fcns = {"nse": calculate_nse,
                      "kge":  calculate_kge,
                      "bias": calculate_bias,
                      "rmse": calculate_rms,
                      }
        
        metrics = {"valid": {},
                   "test" : {},
                   "valid_peak": {},
                   "test_peak" : {},
            }
        
        # initalize metrics
        for key in metric_fcns.keys():
            for on_set in metrics.keys():
                metrics[on_set][key] = []

        total_num_of_folds = len(data_model.cross_sets.keys())
        for num, cross_set in enumerate(data_model.cross_sets.keys()):
            # logging
            TensorBoardCallback = tf.keras.callbacks.TensorBoard(
                os.path.join(current_log_path, f"fold_{num:02d}"), 
                write_graph  = False,
                write_images = False,
                histogram_freq=None)
    

            print(f"processing cross_set {cross_set} -------------------------------")
            X_train, y_train = data_model.getDataSet(data_model.cross_sets[cross_set]["train"], scale=True, shuffle=shuffle)
            X_valid, y_valid = data_model.getDataSet(data_model.cross_sets[cross_set]["valid"], scale=True)
            
            # get simulation and measured values 
            _, _, yidx = data_model.sets[data_model.cross_sets[cross_set]["valid"]]

            # build model
            K.clear_session()
            model = self.hypermodel.build(hp)
            
            # training on training set
            print("train model")
            # reset learning rate to inital value
            K.set_value(model.optimizer.learning_rate, hp["lr"])
            model.fit(X_train, y_train, 
                        epochs     = epochs, 
                        batch_size = hp["batch_size"], 
                        validation_data = (X_valid, y_valid), 
                        callbacks = callbacks + [TensorBoardCallback], 
                        verbose = 1,
                        workers = 4,
                        use_multiprocessing=True,)
            
            # eval on validation set
            y_pred_valid = model.predict(X_valid,
                                        batch_size = hp["batch_size"], 
                                        workers = 4,
                                        use_multiprocessing=True)
            
            for key in metric_fcns.keys():
                losses = evaluate_multistep(y_valid, 
                                            y_pred_valid, 
                                            metric_fcns[key])
                metrics["valid"][key].append(losses)
                
            del y_pred_valid, losses, X_train, y_train
               
            # load new data
            X_train_valid, y_train_valid = data_model.getDataSet(data_model.cross_sets[cross_set]["train_valid"][-1:], scale=True, shuffle=shuffle) 
            
            print("retrain model with new data")
            # reset learning rate to half of initial value
            K.set_value(model.optimizer.learning_rate, hp["lr"]/2)
            
            # continue training with validation set   
            model.fit(X_train_valid, y_train_valid, 
                      epochs     = hp["retrain_epochs"], 
                      batch_size = hp["batch_size"],
                      callbacks = TensorBoardCallback,
                      verbose    = 1,
                      workers    = 4,
                      use_multiprocessing=True)
            
            del X_train_valid, y_train_valid
            
            # evaluate on testing set
            print("evaluate model performence")
            
            # load new data
            X_test,  y_test  = data_model.getDataSet(data_model.cross_sets[cross_set]["test"], scale=True, shuffle=False) 
            
            # get simulation and measured values 
            _, _, yidx = data_model.sets[data_model.cross_sets[cross_set]["test"]]
            
            y_pred_test = model.predict(X_test, 
                                        batch_size = hp["batch_size"], 
                                        workers = 4,
                                        use_multiprocessing=True)

            for key in metric_fcns.keys():
                losses = evaluate_multistep(y_test, 
                                            y_pred_test, 
                                            metric_fcns[key])
                metrics["test"][key].append(losses)
                
            del losses 
            
            # save fold model
            if save_fold_models:
                self.save_model_fold(trial, num, model)

            # save fold forecast data
            y_pred_test = y_pred_test
            if save_fold_prediction:
                np.savetxt(os.path.join(current_log_path, f"pred_fold_{num}.txt"),
                          y_pred_test, delimiter=",")
            
            # plotting
            if plot_fold_rst:
                fig, ax = plt.subplots(1,1,figsize=(16,9))
                ax.set_title(f'fold {num}')
                ax.set_ylabel('q')
                ax.set_xlabel('step')
                
                tt_test = np.arange(len(y_test))
                
                pred00 = y_pred_test[:,0].reshape(-1,1)
                pred95 = y_pred_test[:,-1].reshape(-1,1)
                
                ax.plot(tt_test,    y_test[:,0,0], 'gray')
                ax.plot(tt_test,    pred00, 'blue', label="1-step-forecast")
                ax.plot(tt_test+95, pred95,'green', label="96-step-forecast")
                
                ax.legend()
                fig.show()

                fig.savefig(os.path.join(self.tb_log_path, 
                                         "logs", 
                                         "trial_"+f"{trial.trial_id}",
                                         f"eval_fold_{num}.png"), 
                            dpi = 120,
                            )
                plt.pause(0.001)
                
                idx_peak = np.argmax(y_test[:,0])
                ax.set_xlim((tt_test[idx_peak]-24, tt_test[idx_peak]+24))
                
                fig.show()
                fig.savefig(os.path.join(self.tb_log_path, 
                                         "logs", 
                                         "trial_"+f"{trial.trial_id}",
                                         f"eval_fold_{num}_peak.png"), 
                            dpi = 120,
                            )
                
                plt.pause(0.001)
                plt.close('all')
                del fig
            
            # delete variables
            del X_test, y_test, y_pred_test
            
            # write for tensorboard
            with tf.summary.create_file_writer(current_log_path).as_default():
                for key in ["kge", "nse"]:
                    tf.summary.scalar(f'{key}_fold_{num}',  np.mean(metrics["test"][key][num]), step=1)
                    
            # verbose
 
            print_metrics_valid = [np.mean(metrics["valid"][x][num]) for x in ["kge", "nse", "bias"]]
            print_metrics_test  = [np.mean(metrics["test" ][x][num]) for x in ["kge", "nse", "bias"]]
            
            print(f"valid kge - nse - bias: {[f'{x:6.4f}' for x in print_metrics_valid]}")
            print(f"test  kge - nse - bias: {[f'{x:6.4f}' for x in print_metrics_test]}")

        obj_losses = np.mean([np.mean(metrics["valid"]["kge"][x]) + np.mean(metrics["valid"]["nse"][x]) for x in range(total_num_of_folds)])
        
        print(f"objective loss: {2 - obj_losses}")
        
        # save loss values
        with open(os.path.join(current_log_path, "metrics.txt"), "w+") as f:
            json.dump(metrics, f)
        
        # write for tensorboard
        num_trainable     = int(np.sum([p.numpy().size for p in model.trainable_weights]))
        num_non_trainable = int(np.sum([p.numpy().size for p in model.non_trainable_weights]))
        with tf.summary.create_file_writer(current_log_path).as_default():
            tf.summary.scalar("trial_id",      np.float64(trial.trial_id), step=1)
            tf.summary.scalar("trainable",     num_trainable, step=1)
            tf.summary.scalar("non_trainable", num_non_trainable, step=1)
            for key in ["kge", "nse"]:
                for on_set in ["valid", "test"]:
                    m = np.mean(metrics[on_set][key])
                    tf.summary.scalar(f"{key}_{on_set}",  m, step=1)
                    print(f"{on_set}: {m:6.4f}")
        
        # save final model
        self.save_model(trial, model)
        
        return 2 - obj_losses