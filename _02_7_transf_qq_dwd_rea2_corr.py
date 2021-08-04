
# !/usr/bin/env python.
# -*- coding: utf-8 -*-

'''
TODO: Add docs

'''

# from .cfcoords import translate_coords
# from .datamodels import CDS, ECMWF
import sys

import numpy as np

import os

import numpy as numpy
import pandas as pd

import glob
import matplotlib.pyplot as plt
import tqdm
from _00_additional_functions import (resampleDf,
                                      find_nearest,
                                      transform_to_bools,
                                      build_edf_fr_vals)

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")


path_to_all_rea2_files = r'/home/abbas/Documents/REA2/REA_Pcp'

list_years = np.arange(2013, 2014, 1)

# percentile_level = 0.99

test_for_extremes = False

dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']

os.chdir(path_to_all_rea2_files)
all_grib_files = glob.glob('*.csv')

aggs = ['60min', '120min', '180min', '360min', '720min', '1440min']


def qq_transform_Rea2(stn_id, df_rea, df_dwd_recent, df_dwd_hist):
    xdwd, ydwd = build_edf_fr_vals(df_dwd_recent.values)
    xdwd_hist, ydwd_hist = build_edf_fr_vals(
        df_dwd_hist.values)
    xrea2, yrea2 = build_edf_fr_vals(df_rea.values)

    df_xrea2_tranf = pd.DataFrame(index=df_rea.index,
                                  columns=[stn_id])

    for idx, pcp in zip(df_rea.index, df_rea.values):
        #                 pcp=5
        if pcp > 0.1:
            qrea2 = yrea2[
                np.max(np.where(xrea2 == find_nearest(xrea2, pcp)))]
            pcp_dwd_q_rea2 = xdwd[
                np.max(np.where(ydwd == find_nearest(ydwd, qrea2)))]

            q_dwd_q_rea2_hist = ydwd_hist[
                np.max(np.where(xdwd_hist == find_nearest(
                    xdwd_hist, pcp_dwd_q_rea2)))]

            pcp_rea2_trans = xrea2[
                np.max(np.where(
                    yrea2 == find_nearest(yrea2, q_dwd_q_rea2_hist)))]
            df_xrea2_tranf.loc[idx, stn_id] = pcp_rea2_trans
#                     break
        else:
            df_xrea2_tranf.loc[idx, stn_id] = pcp

    return df_xrea2_tranf


# =============================================================================
# 
# =============================================================================
for _year in list_years:
    print(_year)
    start_year = '01-01-%s 00:00:00' % _year
    end_year = '31-12-%s 23:00:00' % _year
    pcp_Rea = [df for df in all_grib_files if str(_year) in df]
    in_df_rea2 = pd.read_csv(pcp_Rea[0], sep=';',
                             index_col=0,
                             parse_dates=True,
                             infer_datetime_format=True)
    
    in_df_rea2[in_df_rea2 < 0] = 0
    # read data and get station ids and coords
    empty_df_rea2 = pd.DataFrame(
        index=in_df_rea2.index,
        columns=in_df_rea2.columns,
        data=np.full(shape=(in_df_rea2.shape), fill_value=np.nan))
    
    dwd_pcp = dwd_hdf5.get_pandas_dataframe_between_dates(
        dwd_ids,
        event_start=start_year,
        event_end=end_year)
    dwd_pcp_hourly = resampleDf(dwd_pcp, 'H')
    # for temp_agg in (aggs):
        # print(temp_agg)
        # dwd_pcp_res = resampleDf(dwd_pcp, temp_agg)
        # in_df_rea2_res = resampleDf(in_df_rea2, temp_agg)

    for _ii in tqdm.tqdm(range(len(dwd_ids))):
        #             print(_ii, '/', len(dwd_ids))
        stn_id = dwd_ids[_ii]
        # dwd pcp historical
        print(stn_id)
        dwd_pcp_hist = dwd_hdf5.get_pandas_dataframe(stn_id)
        dwd_pcp_hist_hourly = resampleDf(dwd_pcp_hist, 'H').dropna(how='any')
        #dwd_pcp_hist_hourly

        df_dwd1 = dwd_pcp_hourly.loc[:, stn_id].dropna(how='any')

        df_rea1 = in_df_rea2.loc[:, stn_id].dropna(how='any')

        df_rea1[df_rea1 < 0] = 0
        
        if df_dwd1.size > 0:

            try:
                df_rea1_tranf = qq_transform_Rea2(
                stn_id,
                df_rea1,
                df_dwd1,
                dwd_pcp_hist_hourly)
            except Exception:
                print('error')
            empty_df_rea2.loc[df_rea1_tranf.index, stn_id] = df_rea1_tranf.values
    
    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=100)
    plt.scatter( in_df_rea2.values, empty_df_rea2.values,
                c='k', marker='o', alpha=0.5, s=40)
    plt.xlabel('REA2 Original [mm/60min]', fontsize=14)
    plt.ylabel('REA2 QQ-Transformed [mm/60min]', fontsize=14)
    plt.grid(alpha=0.5)
    #plt.plot([0, max(np.nanmax(empty_df_rea2.values), np.nanmax(in_df_rea2.values))],
    #         [0, max(np.nanmax(empty_df_rea2.values), np.nanmax(in_df_rea2.values))],
    #         c='r', linestyle='-.')
    
    plt.savefig(os.path.join(
        path_to_all_rea2_files, 'qq_pcp_%s.png' % _year))
    plt.close()
    
    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=100)
    plt.scatter( in_df_rea2.values, empty_df_rea2.values,
                c='k', marker='o', alpha=0.5, s=40)
    plt.xlabel('REA2 Original [mm/60min]', fontsize=14)
    plt.ylabel('REA2 QQ-Transformed [mm/60min]', fontsize=14)
    plt.plot([0, max(np.nanmax(empty_df_rea2.values), np.nanmax(in_df_rea2.values))],
             [0, max(np.nanmax(empty_df_rea2.values), np.nanmax(in_df_rea2.values))],
             c='r', linestyle='-.')

    plt.savefig(os.path.join(
        path_to_all_rea2_files, 'qq_pcp_%s2.png' % _year))
    plt.close()
    
    empty_df_rea2.to_csv(os.path.join(
        path_to_all_rea2_files, 'qq_pcp_%s.csv' % _year),
                         sep=';')
    # break