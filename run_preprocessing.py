from ForecastModel.utils.preprocessing import CreateIndices

import pandas as pd
import numpy as np

OUT_DATA_PATH    = r"data\dummy_data.csv"
OUT_INDICES_PATH = r"data\indices" 

# create random dummy data
# for testing only
print("create random dummy data")
df = pd.DataFrame()
df["time"]      = pd.date_range("1-1-2020","31-12-2020", freq="15min")
df              = df.set_index("time")

df["pmean"]     = np.random.lognormal(0.03,1.2, df.shape[0])/10
df.loc[df["pmean"] < 4, "pmean"] = 0
df["pmean"]     = df["pmean"].rolling(20).mean().bfill()*5
df["pmax"]      = df["pmean"] * 4

df["tmax"]      = 9.96 - 15.3*np.cos(2*np.pi*np.arange(0,df.shape[0]) / (4*24*365) - np.pi/7) + np.random.normal(0,2.1, df.shape[0])
df["tmean"]     = 7.24 - 13.2*np.cos(2*np.pi*np.arange(0,df.shape[0]) / (4*24*365) - np.pi/7) + np.random.normal(0,0.7, df.shape[0])
df["tmin"]      = 4.18 - 16.4*np.cos(2*np.pi*np.arange(0,df.shape[0]) / (4*24*365) - np.pi/7) + np.random.normal(0,1.9, df.shape[0])

df["qmeasval"]  = np.random.lognormal(0.87,0.63, df.shape[0])

df["qsim"]  = df["qmeasval"] + np.random.normal(1.2,0.8, df.shape[0])

# output
df.to_csv(OUT_DATA_PATH)

# create indices
ci = CreateIndices(OUT_DATA_PATH, out_path=OUT_INDICES_PATH)
ci.create(n_sets = 4, 
        hincast_lengths = [96, 2*96], 
        forecast_len = 96, 
        target_len = 96,
        )