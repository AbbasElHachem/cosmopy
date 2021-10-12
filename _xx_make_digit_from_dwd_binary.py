#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:18:42 2021

@author: abbas
"""

# !/usr/bin/env python.
# -*- coding: utf-8 -*-

'''
TODO: Add docs

'''

# from .cfcoords import translate_coords
# from .datamodels import CDS, ECMWF
import sys
import xarray as xr
import numpy as np
import cf2cdm
import cfgrib
import os
import netCDF4
import numpy as numpy
import pandas as pd
import rasterio
import shapefile as shp  # Requires the pyshp package
import matplotlib
import matplotlib.pyplot as plt
import glob
from scipy.spatial import distance
from scipy.stats import spearmanr as spr
from scipy.stats import pearsonr as prs
import tqdm
from _00_additional_functions import resampleDf, get_cdf_part_abv_thr

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")


dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']


# transform to boolean
dwd_pcp = dwd_hdf5.get_pandas_dataframe(
        dwd_ids[0])


dwd_pcp_daily = resampleDf(dwd_pcp, '1440min')

thr = 1

dwd_pcp_daily_boolean = dwd_pcp_daily.copy()

dwd_pcp_daily_boolean[dwd_pcp_daily_boolean >= thr] = 1
dwd_pcp_daily_boolean[dwd_pcp_daily_boolean < thr] = 0

dwd_pcp_daily_boolean_test = dwd_pcp_daily_boolean.values[:6].ravel().astype(int)
# Functionally:

num = str(''.join(map(str,dwd_pcp_daily_boolean_test)))

decimal = 0
for digit in num:
    decimal = decimal*2 + int(digit)

print(decimal)

int(num)
