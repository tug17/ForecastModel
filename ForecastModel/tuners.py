#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27

@authors: manuel, sebastian
"""
import os
import tensorflow as tf

import numpy as np
import keras_tuner

from tensorboard.plugins.hparams import api as tb_hp

from datetime import datetime
import matplotlib.pyplot as plt

from ForecastModel.utils.metrics import evaluate_multistep, calculate_bias, calculate_kge, calculate_nse

# %%
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
        
    def run_trial(self, trial, data_model, verbose, epochs, callbacks, cross_indices_path, tb_log_path, plot_fold_rst=True, save_fold_models=False, save_fold_prediction=True):
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
        #data_model.main(os.path.join(cross_indices_path, f"cross_indices_{hindcast_length}.json"), verbose)
        
        val_losses_kge = []
        val_losses_nse = []
        val_losses_bias= []
        metric_nse = []
        metric_kge = []
        metric_bias= []
        for num,key in enumerate(data_model.cross_sets.keys()):
            # logging
            TensorBoardCallback = tf.keras.callbacks.TensorBoard(
                os.path.join(current_log_path, f"fold_{num:02d}"), 
                write_graph  = False,
                write_images = False,
                histogram_freq=None)
    

            print(f"processing cross_set {key} -------------------------------")
            X_train, y_train = data_model.getDataSet(data_model.cross_sets[key]["train"], scale=True) 
            X_valid, y_valid = data_model.getDataSet(data_model.cross_sets[key]["valid"], scale=True) 
            X_test,  y_test  = data_model.getDataSet(data_model.cross_sets[key]["test"], scale=True) 
            X_train_valid, y_train_valid = data_model.getDataSet(data_model.cross_sets[key]["train_valid"], scale=True) 
            
            # build model
            tf.keras.backend.clear_session()
            model = self.hypermodel.build(hp)
            
            # # training 
            # print("training with train set")
            model.fit(X_train, y_train, 
                        epochs     = epochs, 
                        batch_size = hp["batch_size"], 
                        validation_data = (X_valid, y_valid), 
                        callbacks = callbacks + [TensorBoardCallback], 
                        verbose = 0)
            
            model.fit(X_train_valid, y_train_valid, 
                      epochs     = hp["retrain_epochs"], 
                      batch_size = hp["batch_size"],
                      callbacks = TensorBoardCallback,
                      verbose    = 0)
            
            # save fold model
            if save_fold_models:
                self.save_model_fold(trial, num, model)
            
            # evaluate model
            print("evaluate model performence")
            y_pred = model.predict(X_test)
            
            print(y_test.shape)
            print(y_pred.shape)
            
            losses_nse = evaluate_multistep(y_test[:,hp["osc_length"]:,:], y_pred[:,hp["osc_length"]:], calculate_nse)
            losses_kge = evaluate_multistep(y_test[:,hp["osc_length"]:,:], y_pred[:,hp["osc_length"]:], calculate_kge)
            losses_bias= evaluate_multistep(y_test[:,hp["osc_length"]:,:], y_pred[:,hp["osc_length"]:], calculate_bias)

            print(f"metrics NSE, KGE, bias: {np.mean(losses_nse):6.4f}, {np.mean(losses_kge):6.4f}, {np.mean(losses_bias):6.4f}")
            val_losses_kge.append(np.mean(losses_kge))
            val_losses_nse.append(np.mean(losses_nse))
            val_losses_bias.append(np.mean(losses_bias))
            metric_nse.append(losses_nse)
            metric_kge.append(losses_kge)
            metric_bias.append(losses_bias)
            
            # save fold forecast data
            y_pred = y_pred[:,hp["osc_length"]:]
            if save_fold_prediction:
                np.savetxt(os.path.join(current_log_path, f"pred_fold_{num}.txt"),
                          y_pred, delimiter=",")
            
            # plotting
            if plot_fold_rst:
                fig = plt.figure(figsize=(16,9))
                plt.title(f'fold {key}')
                plt.ylabel('q')
                plt.xlabel('step')
                
                tt_test  = len(y_train_valid) + np.arange(len(y_test))
                tt_train_valid = np.arange(len(y_train_valid))
                
                pred00 = y_pred[:,0].reshape(-1,1)
                pred95 = y_pred[:,-1].reshape(-1,1)
                
                plt.plot(tt_train_valid, y_train_valid[:,0],'gray', label="ground truth")
                plt.plot(tt_test,        y_test[:,0],       'gray')
                plt.plot(tt_test,    pred00, 'blue', label="1-step-forecast")
                plt.plot(tt_test+95, pred95,'green', label="96-step-forecast")
                
                plt.xlim(tt_train_valid[-1]-96*7, tt_test[-1]+95)
                plt.legend()
                # plt.show()
                
                fig.savefig(os.path.join(self.tb_log_path, 
                                         "logs", 
                                         "trial_"+f"{trial.trial_id}",
                                         f"eval_fold_{num}.png"), 
                            dpi = 120,
                            )
                plt.close(fig)
            
            # write for tensorboard
            with tf.summary.create_file_writer(current_log_path).as_default():
                tf.summary.scalar(f'kge_fold_{num}',  np.mean(losses_kge), step=1)
                tf.summary.scalar(f'nse_fold_{num}',  np.mean(losses_nse), step=1)
                tf.summary.scalar(f'bias_fold_{num}', np.mean(losses_bias),step=1)
            
            
         # calculate total loss
        val_losses = val_losses_kge
        total_loss = 1 - np.mean(val_losses)
        
        # save loss values
        np.savetxt(os.path.join(current_log_path, "val_losses.txt"),
                      np.array([total_loss] + val_losses), delimiter=",")
        np.savetxt(os.path.join(current_log_path, "metric_nse.txt"),
                      np.array(metric_nse), delimiter=",")
        np.savetxt(os.path.join(current_log_path, "metric_kge.txt"),
                      np.array(metric_kge), delimiter=",")
        np.savetxt(os.path.join(current_log_path, "metric_bias.txt"),
                      np.array(metric_bias), delimiter=",")
        
        # write for tensorboard
        num_trainable     = int(np.sum([p.numpy().size for p in model.trainable_weights]))
        num_non_trainable = int(np.sum([p.numpy().size for p in model.non_trainable_weights]))
        with tf.summary.create_file_writer(current_log_path).as_default():
            tf.summary.scalar("trial_id",      np.float64(trial.trial_id), step=1)
            tf.summary.scalar("trainable",     num_trainable, step=1)
            tf.summary.scalar("non_trainable", num_non_trainable, step=1)
            tf.summary.scalar('kge_total',  np.mean(val_losses_kge), step=1)
            tf.summary.scalar('nse_total',  np.mean(val_losses_nse), step=1)
            tf.summary.scalar('bias_total', np.mean(val_losses_bias),step=1)
        
        # save final model
        self.save_model(trial, model)
        
        return total_loss

#%%

class MyARIMATuner(keras_tuner.BayesianOptimization):
    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        with tf.summary.create_file_writer(os.path.join(self.tb_log_path, r"logs/trial_"+f"{trial.trial_id}")).as_default():
            score = trial.score
            hparams = trial.hyperparameters.get_config()['values']
            tf.summary.scalar('score', score, step=1)
            tb_hp.hparams(hparams)
        
    def save_model(self, trial, model):
        model.save(os.path.join(self.tb_log_path, "hp", f"trial_{trial.trial_id}", "model.keras"))
        
    def save_model_fold(self, trial, fold_id, model):
        model.save(os.path.join(self.tb_log_path, "hp", f"trial_{trial.trial_id}", f"model_fold_{fold_id}.keras"))
        
    def run_trial(self, trial, data_model, verbose, epochs, callbacks, cross_indices_path, tb_log_path, plot_fold_rst=True, save_fold_models=False, save_fold_prediction=True):
        print(trial.trial_id)
        # set tb_log_path
        self.tb_log_path = tb_log_path
        
        current_log_path = os.path.join(tb_log_path, r"logs/trial_"+f"{trial.trial_id}")
        
        # load hyperparameters
        hp = trial.hyperparameters
        
        # get data model
        print(hp)
        hindcast_length = hp["hindcast_length"]
        data_model.main(os.path.join(cross_indices_path, f"cross_indices_{hindcast_length}.pckl"), verbose)
        #data_model.main(os.path.join(cross_indices_path, f"cross_indices_{hindcast_length}.json"), verbose)
        
        val_losses_kge = []
        val_losses_nse = []
        val_losses_bias= []
        metric_nse = []
        metric_kge = []
        metric_bias= []
        for num,key in enumerate(data_model.cross_sets.keys()):
            # logging
            TensorBoardCallback = tf.keras.callbacks.TensorBoard(
                os.path.join(current_log_path, f"fold_{num:02d}"), 
                write_graph  = False,
                write_images = False,
                histogram_freq=None)
    

            print(f"processing cross_set {key} -------------------------------")
            #X_train, y_train = data_model.getDataSet(data_model.cross_sets[key]["train"], scale=True) 
            #X_valid, y_valid = data_model.getDataSet(data_model.cross_sets[key]["valid"], scale=True) 
            X_test,  y_test  = data_model.getDataSet(data_model.cross_sets[key]["test"], scale=True) 
            #X_train_valid, y_train_valid = data_model.getDataSet(data_model.cross_sets[key]["train_valid"], scale=True) 
            
            model = self.hypermodel.build(hp)

            # evaluate model
            print("evaluate model performence")
            y_pred = model.predict(X_test)  
            
            print(y_pred.shape)
            print(y_test.shape)
        
            losses_nse = evaluate_multistep(y_test[:,:,:], y_pred[:,:], calculate_nse)
            losses_kge = evaluate_multistep(y_test[:,:,:], y_pred[:,:], calculate_kge)
            losses_bias= evaluate_multistep(y_test[:,:,:], y_pred[:,:], calculate_bias)            
            
            print(f"metrics NSE, KGE, bias: {np.mean(losses_nse):6.4f}, {np.mean(losses_kge):6.4f}, {np.mean(losses_bias):6.4f}")
            val_losses_kge.append(np.mean(losses_kge))
            val_losses_nse.append(np.mean(losses_nse))
            val_losses_bias.append(np.mean(losses_bias))
            metric_nse.append(losses_nse)
            metric_kge.append(losses_kge)
            metric_bias.append(losses_bias)            

            if save_fold_prediction:
                np.savetxt(os.path.join(current_log_path, f"pred_fold_{num}.txt"),
                          y_pred, delimiter=",")      
                
            # # plotting
            # if plot_fold_rst:
            #     fig = plt.figure(figsize=(16,9))
            #     plt.title(f'fold {key}')
            #     plt.ylabel('q')
            #     plt.xlabel('step')
                
            #     #tt_test  = len(y_train_valid) + np.arange(len(y_test))
            #     #tt_train_valid = np.arange(len(y_train_valid))
                
            #     pred00 = y_pred[:,0].reshape(-1,1)
            #     pred95 = y_pred[:,-1].reshape(-1,1)
                
            #     #plt.plot(tt_train_valid, y_train_valid[:,0],'gray', label="ground truth")
            #     #plt.plot(tt_test,        y_test[:,0],       'gray')
            #     #plt.plot(tt_test,    pred00, 'blue', label="1-step-forecast")
            #     #plt.plot(tt_test+95, pred95,'green', label="96-step-forecast")
                
            #     plt.xlim(tt_train_valid[-1]-96*7, tt_test[-1]+95)
            #     plt.legend()
            #     # plt.show()
                
            #     fig.savefig(os.path.join(self.tb_log_path, 
            #                               "logs", 
            #                               "trial_"+f"{trial.trial_id}",
            #                               f"eval_fold_{num}.png"), 
            #                 dpi = 120,
            #                 )
            #     plt.close(fig)
            
            # write for tensorboard
            with tf.summary.create_file_writer(current_log_path).as_default():
                tf.summary.scalar(f'kge_fold_{num}',  np.mean(losses_kge), step=1)
                tf.summary.scalar(f'nse_fold_{num}',  np.mean(losses_nse), step=1)
                tf.summary.scalar(f'bias_fold_{num}', np.mean(losses_bias),step=1)
            
            
         # calculate total loss
        val_losses = val_losses_kge
        total_loss = 1 - np.mean(val_losses)
        
        # save loss values
        np.savetxt(os.path.join(current_log_path, "val_losses.txt"),
                      np.array([total_loss] + val_losses), delimiter=",")
        np.savetxt(os.path.join(current_log_path, "metric_nse.txt"),
                      np.array(metric_nse), delimiter=",")
        np.savetxt(os.path.join(current_log_path, "metric_kge.txt"),
                      np.array(metric_kge), delimiter=",")
        np.savetxt(os.path.join(current_log_path, "metric_bias.txt"),
                      np.array(metric_bias), delimiter=",")
        
        # write for tensorboard
        # num_trainable     = int(np.sum([p.numpy().size for p in model.trainable_weights]))
        # num_non_trainable = int(np.sum([p.numpy().size for p in model.non_trainable_weights]))
        # with tf.summary.create_file_writer(current_log_path).as_default():
        #     tf.summary.scalar("trial_id",      np.float64(trial.trial_id), step=1)
        #     tf.summary.scalar("trainable",     num_trainable, step=1)
        #     tf.summary.scalar("non_trainable", num_non_trainable, step=1)
        #     tf.summary.scalar('kge_total',  np.mean(val_losses_kge), step=1)
        #     tf.summary.scalar('nse_total',  np.mean(val_losses_nse), step=1)
        #     tf.summary.scalar('bias_total', np.mean(val_losses_bias),step=1)
        
        # save final model
        #self.save_model(trial, model)
        
        return total_loss