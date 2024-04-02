# Hindcast-LSTM-model for improvment of hydrological forecasts
Accompanying code (DOI: 10.5281/zenodo.10907305) for our HESS paper (SUBMITTED) "Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model"

```
Gegenleithner, S., Pirker, M., Dorfmann, C., Kern, R., Schneider, J., 2024. 
Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model. 
```

The manuscript can be found here : [Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model (TO BE UPDATED)](https://github.com/tug17/ForecastModel)

The code in this repository was used to produce and train all models in our manuscript.


## Content of the repository

py-files
- `environment.yml` contains installed packeges and dependencies
- `run_arima.py` python file to run ARIMA model calibration and prediction
- `run_preprocessing.py` python file for preprocessing and creating random data
- `run_tuner.py` python file to train our ML models
- `notebook_*` notebooks used to create paper figures and tables as well as auxilary plots
- `ForecastModel/` contains the entire code to create, train and tune ARIMA and LSTM models
   - `ForecastModel/models.py` model architectures code 
   - `ForecastModel/tuners.py` tuner code 
   - `ForecastModel/data/` contains code for data model to load samples during training
   - `ForecastModel/utils/` contains code for metrics and loss calculations, as well as post- and preprocessing functions
- `data/` containes dataset and sequence index arrays
- `tb/` contains tuner logs and hyperparameters for tensorboard
- `rst/` contains final trained models, fold predictions and evaluated metrics
   - `rst/ARIMA` contains ARIMA result files
   - `rst/HLSTM-PBHM` contains HLSTM-PBHM result files and model save files for each fold
   - `rst/HLSTM` contains HLSTM result files and model save files for each fold
   
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

### Data required
All data is published and made publicly available via zenodo.org (DOI: 10.5281/zenodo.10907245)
In order to run the paper code and notebooks, download the `Dataset.csv` and place it in `data/`.
`model_results.zip` have to be downloaded and unziped into `rst/`.

### Run locally
Activate conda environment:

```
conda activate tf2
```

During pre-processing index arrays are created at `data/indices`, which are later used do build the sequences for training.

```
python run_preprocessing.py
```
Tuning process is started and logs for tensorboard as well as the fold models are saved to `tb/`.

```
python run_tuner.py
```

### Run notebooks
Notebooks can be run in the same environment.

## Citation
If you use any of this code in your experiments, please make sure to cite the following publication

```
author = {Gegenleithner, S., Manuel Pirker, M., Dorfmann, C., Kern, R., and Schneider, J.},
title = {(SUBMITTED) Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model},
year = {2024},
}
```

