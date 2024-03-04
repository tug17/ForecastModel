# Hindcast-LSTM-model for improvment of hydrological forecasts
Accompanying code for our HESS paper "Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model"

```
Gegenleithner, S., Manuel Pirker, M., Dorfmann, C., Kern, R., and Schneider, J.: Long Short-Term Memory 
Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model. 
not published.
```

The manuscript can be found here : [Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model (TO BE UPDATED)](https://github.com/tug17/ForecastModel)

The code in this repository was used to produce and train all models in our manuscript.


## Content of the repository

- `run_preprocessing.py` python file for preprocessing and creating random data
- `run_tuner.py` python file to train our models
- `environment.yml` contains installed packeges and dependencies
- `ForecastModel/` contains the entire code to create, train and tune ARIMA and LSTM models
   - `ForecastModel/models.py` model architectures code 
   - `ForecastModel/tuners.py` tuner code 
   - `ForecastModel/data/` contains code for data model to load samples during training
   - `ForecastModel/utils/` contains code for metrics and loss calculations, as well as post- and preprocessing functions
- `data/` containes dataset and sequence index arrays
- `tb/` contains tuner logs and hyperparameters for tensorboard
- `final_models/` contains trained models and fold predictions 

## Setup to run the code locally

Download this repository either as zip-file or clone it to your local file system by running

```
git clone git@github.com:tug17/ForecastModel.git
```

### Setup Python environment
Within this repository we provide a environment file (`environment.yml`) that can be used with Anaconda or Miniconda to create an environment with all packages needed.
Build on Tensorflow 2.10 runs on Windows 10 with a CUDA capable NVIDIA GPU. 

```
conda env create -f environment.yml
```

## Data needed
As original data cannot be published, dummy data will be created randomly during preprocessing.

## Run locally

Activate conda environment:

```
conda activate tf2
```

During pre-processing random data is generated and saved to data/. The created index arrays at data/indices are later used do build the sequences for training.

```
python run_preprocessing.py
```
Tuning process is started and logs and fold models are saved to tb/.

```
python run_tuner.py
```


## Citation

If you use any of this code in your experiments, please make sure to cite the following publication (TO BE UPDATED)

```
author = {Gegenleithner, S., Manuel Pirker, M., Dorfmann, C., Kern, R., and Schneider, J.},
title = {Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model},
}
```

