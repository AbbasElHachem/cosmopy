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
    r'/run/media/abbas/EL Hachem 2019/REA6/Analysis/cross_corr')
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)
    
list_years = np.arange(1995, 2020, 1)

# percentile_level = 0.99

test_for_extremes = True
calc_qq_corr = False
calc_cross_corr = False


aggs = ['60min', '120min', '180min', '360min', '720min', '1440min']

percentile_levels = [0.99, 0.98, 0.97, 0.95, 0.93, 0.92]

# =============================================================================
# read data 
# =============================================================================
print('Raeding rea6 data')
in_df_rea6 = pd.read_csv(path_to_rea6_files, sep=';',
                             index_col=0, engine='c')

in_df_rea6.index = pd.to_datetime(in_df_rea6.index, format='%Y-%m-%d %H:%M:%S')
#in_df_rea6 = in_df_rea6 * 3600
#in_df_rea6.dropna(how='all')

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
# 
# =============================================================================
    
for temp_agg, percentile_level in zip(aggs, percentile_levels):
    print(temp_agg)
    
    df_distance_corr = pd.DataFrame(index=dwd_ids_hannover)
    
    for stn_id in tqdm.tqdm(dwd_ids_hannover):
        #             print(_ii, '/', len(dwd_ids))
        
        dwd_pcp = dwd_hdf5_de.get_pandas_dataframe(stn_id).dropna()
        in_df_rea6_stn = in_df_rea6.loc[:, stn_id].dropna()
        
        cmn_idx = dwd_pcp.index.intersection(in_df_rea6_stn.index)
        #break

        df_dwd1 = resampleDf(dwd_pcp.loc[cmn_idx,:], temp_agg)
        df_rea1 = resampleDf(in_df_rea6_stn.loc[cmn_idx], temp_agg)
        
        if df_dwd1.size > 0:
            if test_for_extremes:
                df_dwd1 = transform_to_bools(df_dwd1, percentile_level)
                df_rea1 = transform_to_bools(df_rea1, percentile_level)
                
                spr_corr_dwd_rea = spr(df_dwd1.values.ravel(), df_rea1.values.ravel())[0]
                prs_corr_dwd_rea = prs(df_dwd1.values.ravel(), df_rea1.values.ravel())[0]
                
                
            else:
                spr_corr_dwd_rea = spr(df_dwd1.values.ravel(), df_rea1.values.ravel())[0]
                prs_corr_dwd_rea = prs(df_dwd1.values.ravel(), df_rea1.values.ravel())[0]


        df_distance_corr.loc[stn_id,
                             'prs_corr_dwd_rea'] = prs_corr_dwd_rea
        df_distance_corr.loc[stn_id,
                             'spr_corr_dwd_rea'] = spr_corr_dwd_rea


    # all stns
    #======================================================================

    prs_corr_dwd = df_distance_corr.loc[:, 'prs_corr_dwd_rea']
    spr_corr_dwd = df_distance_corr.loc[:, 'spr_corr_dwd_rea']


    #======================================================================
    # pearson corr
    #======================================================================
    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=200)
    plt.scatter(range(len(dwd_ids_hannover)),
                prs_corr_dwd, c='b', label='DWD-REA', marker='D',
                alpha=0.75)

    plt.grid(alpha=0.5)
    plt.legend(loc=0)
    plt.xlabel('Station ID')
    plt.ylabel('Pearson Correlation [mm/%s]' % temp_agg)
    if test_for_extremes:
        plt.ylabel('Indicator Correlation [mm/%s]' % temp_agg)
    plt.ylim([-0.01, 1.01])

    if test_for_extremes:
        plt.savefig(os.path.join(
            out_save_dir,
            #             r'analysis',
            r'indic_corr_pair_%d_rea_dwd_%s.png' % (percentile_level, 
                                                      temp_agg)))
    else:
        plt.savefig(os.path.join(
            out_save_dir,
            #             '/analysis',
            r'prs_corr_pairs_rea_dwd_%s.png' % (temp_agg)))
    plt.close()
    
    #======================================================================
    # spearman corr
    #======================================================================
    if not test_for_extremes:
        plt.ioff()
        plt.figure(figsize=(12, 8), dpi=200)
        plt.scatter(range(len(dwd_ids_hannover)),
                    spr_corr_dwd, c='b', label='DWD-REA', marker='D',
                    alpha=0.75)
    
        plt.grid(alpha=0.5)
        plt.legend(loc=0)
        plt.xlabel('Station ID')
        plt.ylabel('Spearman Correlation [mm/%s]' % temp_agg)
        plt.title('Spearman Correlation neighboring DWD-REA pairs %s' % temp_agg)
        plt.ylim([-0.01, 1.01])
    

        plt.savefig(os.path.join(
                out_save_dir,
                #             '/analysis',
                r'spr_corr_pairs_rea_dwd_%s.png' % (temp_agg)))
        plt.close()