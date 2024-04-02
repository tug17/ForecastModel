#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
from ForecastModel.utils.preprocessing import CreateIndices

import pandas as pd
import numpy as np

#############################
#         Init
#############################
OUT_DATA_PATH    = r"data\Dataset.csv"
OUT_INDICES_PATH = r"data\indices" 

#############################
#         Main
#############################
if __name__ == "__main__":
    # create indices
    ci = CreateIndices(OUT_DATA_PATH, out_path=OUT_INDICES_PATH)
    ci.create(n_sets = 7, 
            hincast_lengths = [96], 
            forecast_len = 96, 
            target_len = 96,
            )