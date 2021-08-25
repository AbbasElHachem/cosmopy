
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

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'axes.labelsize': 18})

from read_hdf5 import HDF5

path_dwd_data = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")

out_save_dir = (
    r'/run/media/abbas/EL Hachem 2019/REA6/Analysis/hourly_yearly_cycle')
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)

path_to_all_rea6_files = r'/run/media/abbas/EL Hachem 2019/REA6/Extracted_Hannover'

list_years = np.arange(1995, 2020, 1)

percentile_level = 0.99

test_for_extremes = True

dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']

os.chdir(path_to_all_rea6_files)
all_grib_files = glob.glob('*.csv')

aggs = ['60min', '120min', '180min', '360min', '720min', '1440min']

for _year in list_years:
    print(_year)
    start_year = '01-01-%s 00:00:00' % _year
    end_year = '31-12-%s 23:00:00' % _year
    pcp_Rea = [df for df in all_grib_files if str(_year) in df]
    
    in_df_rea2 = pd.read_csv(pcp_Rea[0], sep=';',
                             index_col=0,
                             parse_dates=True,
                             infer_datetime_format=True)
    # read data and get station ids and coords

    dwd_pcp = dwd_hdf5.get_pandas_dataframe_between_dates(
        dwd_ids,
        event_start=start_year,
        event_end=end_year).dropna(how='all', axis=1)
#     for temp_agg in aggs:

    dwd_monthly_vals = dwd_pcp.groupby(
        [dwd_pcp.index.month],
        dropna=False).sum()

    rea2_monthly_vals = in_df_rea2.groupby(
        [in_df_rea2.index.month],
        dropna=False).sum()
    sum_mon = [4, 5, 6, 7, 8]

    dwd_pcp_summer = dwd_pcp[dwd_pcp.index.month.isin(sum_mon)]
    rea_pcp_summer = in_df_rea2[in_df_rea2.index.month.isin(sum_mon)]
    dwd_hourly_vals_max = dwd_pcp_summer.groupby(
        [dwd_pcp_summer.index.hour],
        dropna=False).max()

    rea2_hourly_vals_max = rea_pcp_summer.groupby(
        [rea_pcp_summer.index.hour],
        dropna=False).max()

#     dwd_hourly_vals_mean = dwd_pcp.groupby(
#         [dwd_pcp.index.hour],
#         dropna=False).mean()

#     rea2_hourly_vals_mean = in_df_rea2.groupby(
#         [in_df_rea2.index.hour],
#         dropna=False).mean()

    plt.ioff()
    plt.figure(figsize=(16, 8), dpi=200)
    for _col in dwd_monthly_vals.columns:
        vals_stn = dwd_monthly_vals.loc[:, _col].values.ravel()

        rea_vals_stn = rea2_monthly_vals.loc[:, _col].values.ravel()
        if np.all(vals_stn) > 0:
            plt.plot(
                dwd_monthly_vals.index,
                vals_stn,
                c='red',
                alpha=0.5)
            plt.plot(
                dwd_monthly_vals.index,
                rea_vals_stn,
                c='grey',
                alpha=0.5)

    plt.grid(alpha=0.5)
    plt.xlabel('Month')
    plt.ylabel('Pcp Sum [mm/Month]')
    plt.title('Yearly cycle \nDWD (red)-REA6 (grey) - Year %s' % _year)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.savefig(os.path.join(
        out_save_dir,
        r'yearly_cycle_%d_rea6_dwd.png'
        % (_year)))

    plt.close()

    plt.ioff()
    plt.figure(figsize=(16, 8), dpi=200)
    for _col in dwd_hourly_vals_max.columns:
        vals_max = dwd_hourly_vals_max.loc[:, _col].dropna().values.ravel()
#         vals_mean = dwd_hourly_vals_mean.loc[:, _col].values.ravel()

        rea_vals_max = rea2_hourly_vals_max.loc[:, _col].values.ravel()
#         rea_vals_mean = rea2_hourly_vals_mean.loc[:, _col].values.ravel()
        if np.all(vals_max) > 0 and vals_max.size > 0:
            plt.plot(
                dwd_hourly_vals_max.index,
                vals_max,
                c='r',
                alpha=0.5)
#             plt.plot(
#                 dwd_hourly_vals_max.index,
#                 vals_mean,
#                 c='orange',
#                 alpha=0.5)
            plt.plot(
                dwd_hourly_vals_max.index,
                rea_vals_max,
                c='grey',
                alpha=0.5)
#             plt.plot(
#                 dwd_monthly_vals.index,
#                 rea_vals_mean,
#                 c='g',
#                 alpha=0.5)
    plt.title('Hourly max-mean - DWD (red) - REA6 (grey)')
    plt.grid(alpha=0.5)
    plt.xlabel('Hour of day')
    plt.ylabel('Pcp Max [mm/hour]')
    plt.title(
        'Diurnal cycle (max/hour/day)\nDWD (red) -REA6 (grey) - Year %s' %
        _year)
    plt.xticks(dwd_hourly_vals_max.index.to_list())
    plt.savefig(os.path.join(
        out_save_dir,
        r'hourly_cycle_%d_rea6_dwd.png'
        % (_year)))

    plt.close()
