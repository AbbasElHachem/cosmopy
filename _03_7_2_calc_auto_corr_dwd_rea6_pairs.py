#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:19:13 2021

@author: abbas
"""


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
from sklearn.preprocessing import quantile_transform
from scipy.stats import linregress
from statsmodels.tsa import stattools
import tqdm
from _00_additional_functions import (resampleDf,
                                      get_cdf_part_abv_thr,
                                      build_edf_fr_vals)

import warnings
warnings.filterwarnings("ignore")

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data_hannover = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")

path_dwd_data_de = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_1min_data_agg_60min_2020_gk3.h5")

path_to_rea6_files = (r'/run/media/abbas/EL Hachem 2019/REA6'
                      r'/Extracted_Hannover/comb_years/rea6_1995_2019.csv')


out_save_dir = (
    r'/run/media/abbas/EL Hachem 2019/REA6/Analysis/rea6_daily')
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)
    
list_years = np.arange(1995, 2020, 1)

# percentile_level = 0.99

test_for_extremes = True
calc_qq_corr = False
calc_cross_corr = False


aggs = ['60min', '120min', '180min', '360min', '720min', '1440min']

aggs = ['1440min']


path_to_rea6_daily = (r'/run/media/abbas/EL Hachem 2019/REA6'
                      r'/Extracted_Hannover_Daily'
                      r'/comb_years/rea6_1995_2019.csv')

# =============================================================================
# read data 
# =============================================================================
print('Raeding rea6 data')
in_df_rea6 = pd.read_csv(path_to_rea6_files, sep=';',
                             index_col=0, engine='c')



# in_df_rea6 = pd.read_csv(path_to_rea6_daily, sep=';',
#                              index_col=0, engine='c')

in_df_rea6.index = pd.to_datetime(in_df_rea6.index, format='%Y-%m-%d %H:%M:%S')

in_df_rea6 = in_df_rea6.round(2)

# =============================================================================
# 
# =============================================================================

dwd_hdf5_hannover = HDF5(infile=path_dwd_data_hannover)
dwd_ids_hannover = dwd_hdf5_hannover.get_all_names()

dwd_hdf5_de = HDF5(infile=path_dwd_data_de)
dwd_ids_de = dwd_hdf5_de.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids_de,
    data=dwd_hdf5_de.get_coordinates(dwd_ids_de)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5_de.get_coordinates(dwd_ids_de)['northing']

print('Reading dwd data')


#dwd_pcp.dropna(how='all').head()

# =============================================================================
# functions
# =============================================================================

def transform_to_bools(df_pcp, perc_thr):
    dwd_cdf_x, dwd_cdf_y = get_cdf_part_abv_thr(
        df_pcp.values.ravel(), -0.1)
    # get dwd ppt thr from cdf
    dwd_ppt_thr_per = dwd_cdf_x[np.where(
        dwd_cdf_y >= perc_thr)][0]
    idx_abv = np.where(df_pcp.values >= dwd_ppt_thr_per)[0]
    idx_below = np.where(df_pcp.values < dwd_ppt_thr_per)[0]
    df_pcp.iloc[idx_abv] = 1
    df_pcp.iloc[idx_below] = 0

    return df_pcp


def fun_calc_cross_corr(vec1, vec2):

    v1_norm = (vec1 - np.mean(vec1)) / (np.std(vec1) * len(vec1))
    v2_norm = (vec2 - np.mean(vec2)) / (np.std(vec2))

    cross_corr = np.correlate(v1_norm, v2_norm)
    return cross_corr


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2

# =============================================================================

# x = 1-D array

# =============================================================================
    
x_ticks = np.arange(0, 13, 1)
for temp_agg in aggs:
    print(temp_agg)
    
    df_auto_corr_dwd = pd.DataFrame(index=dwd_ids_hannover, columns=x_ticks)
    df_auto_corr_rea = pd.DataFrame(index=dwd_ids_hannover, columns=x_ticks)
    
    for stn_id in tqdm.tqdm(dwd_ids_hannover):
        #             print(_ii, '/', len(dwd_ids))
        
        dwd_pcp = dwd_hdf5_de.get_pandas_dataframe(stn_id).dropna()
        in_df_rea6_stn = in_df_rea6.loc[:, stn_id].dropna()
        
        cmn_idx = dwd_pcp.index.intersection(in_df_rea6_stn.index)
        #break

        df_dwd1 = resampleDf(dwd_pcp.loc[cmn_idx,:], temp_agg)
        df_rea1 = resampleDf(in_df_rea6_stn.loc[cmn_idx], temp_agg,
                             closed='right', label='left')
        
        
        # Yield normalized autocorrelation function of number lags
        autocorr_dwd = stattools.acf(df_dwd1, nlags=12, fft=True)
        autocorr_rea = stattools.acf(df_rea1.values.ravel(), nlags=12, fft=True)

        df_auto_corr_dwd.loc[stn_id, :] = autocorr_dwd
        df_auto_corr_rea.loc[stn_id, :] = autocorr_rea


    # all stns

    #======================================================================
    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=200)
    for _ix in df_auto_corr_dwd.index:
        plt.plot(x_ticks,
                df_auto_corr_dwd.loc[_ix,:],
                c='b', marker='.',
                alpha=0.5)
        
        plt.plot(x_ticks,
                df_auto_corr_rea.loc[_ix,:],
                c='r', marker='.',
                alpha=0.5)

    plt.grid(alpha=0.5)
    plt.legend(loc=0)
    plt.xlabel('Lag [%s]' % temp_agg)
    plt.ylabel('Auto-correlation')
    
    plt.savefig(os.path.join(
            out_save_dir,
            #             '/analysis',
            r'auto_corr_pairs_rea_dwd_%s2.png' % (temp_agg)))
    plt.close()
    
