
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

from _00_additional_functions import resampleDf, get_cdf_part_abv_thr

# modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
# sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")


path_to_all_rea2_files = r'/run/media/abbas/EL Hachem 2019/REA_Pcp/pcp_all_Dwd_stns'


path_hourly_extremes = (
    r"/home/abbas/Documents/REA2/rem_timesteps_60min_Hannover_thr_6.00_.csv")
list_years = np.arange(2007, 2012, 1)

percentile_level = 0.99

test_for_extremes = True

dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']

os.chdir(path_to_all_rea2_files)
all_grib_files = glob.glob('*.csv')


in_df_extr = pd.read_csv(path_hourly_extremes,
                         sep=';',
                         index_col=0,
                         parse_dates=True,
                         infer_datetime_format=True)


for _year in list_years:
    print(_year)
    start_year = '01-01-%s 00:00:00' % _year
    end_year = '31-12-%s 23:00:00' % _year
    pcp_Rea = [df for df in all_grib_files if str(_year) in df]
    in_df_rea2 = pd.read_csv(pcp_Rea[0], sep=';',
                             index_col=0,
                             #                              parse_dates=True,
                             #                              infer_datetime_format=True,
                             engine='c')
    in_df_rea2.index = pd.to_datetime(in_df_rea2.index,
                                      format='%Y-%m-%d %H:%M:%S')

    df_extr = in_df_extr.loc[start_year:end_year, :].sum(axis=1).sort_values(
        ascending=False)
    for _Event in df_extr.index:
        start_event = _Event - pd.Timedelta(days=1)
        end_event = _Event + pd.Timedelta(days=1)
        break
        # read data and get station ids and coords

        dwd_pcp = dwd_hdf5.get_pandas_dataframe_between_dates(
            dwd_ids,
            event_start=start_event,
            event_end=end_event).dropna(how='all')
        dwd_pcp_hourly = resampleDf(dwd_pcp, 'H')
        df_rea2 = in_df_rea2.loc[start_event:end_event, dwd_pcp_hourly.columns]

        plt.ioff()
        plt.figure(figsize=(12, 8), dpi=200)

        for ix, _evt in enumerate(dwd_pcp_hourly.index):
            dwd_stns = dwd_pcp_hourly.loc[_evt, :]

            xcoords = dwd_coords_utm32.loc[dwd_stns.index, 'X']
            ycoords = dwd_coords_utm32.loc[dwd_stns.index, 'Y']

            dwd_vals = dwd_pcp_hourly.loc[_evt, :].values.ravel()
            rea2_vals = df_rea2.loc[_evt, :].values.ravel()

            plt.scatter(dwd_vals, rea2_vals, c='b', marker='.', alpha=0.85)
            plt.plot([0, 25], [0, 25], c='grey', marker='_', alpha=0.25)
            plt.grid(alpha=0.25)
            plt.xlabel('DWD [mm/hr]')
            plt.ylabel('REA2 [mm/hr]')
#         plt.savefig(os.path.join(
#             r'/run/media/abbas/EL Hachem 2019/REA_Pcp/analysis/event_based',
#             r'evt_scatter.png'))
#         plt.close()
#             plt.show()
            plt.ioff()
            plt.figure(figsize=(12, 8), dpi=200)
            im = plt.scatter(xcoords, ycoords,
                             c=dwd_vals,
                             vmin=0,
                             cmap=plt.get_cmap('YlGnBu'),
                             label='DWD', marker='x',
                             alpha=0.85)
            plt.colorbar(im, label='mm/hr')
            plt.axis('equal')
            plt.legend(loc=0)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(_evt)
            plt.savefig(os.path.join(
                r'/run/media/abbas/EL Hachem 2019/REA_Pcp/analysis/event_based',
                r'evt_dwd_%d.png' % (ix)))
            plt.close()

            plt.ioff()
            plt.figure(figsize=(12, 8), dpi=200)
            im = plt.scatter(xcoords, ycoords,
                             c=rea2_vals,
                             vmin=0,
                             cmap=plt.get_cmap('YlGnBu'),
                             label='REA2', marker='x',
                             alpha=0.85)
            plt.colorbar(im, label='mm/hr')
            plt.axis('equal')
            plt.xlabel('X')
            plt.legend(loc=0)
            plt.ylabel('Y')
            plt.title(_evt)
            plt.savefig(os.path.join(
                r'/run/media/abbas/EL Hachem 2019/REA_Pcp/analysis/event_based',
                r'evt_rea2_%d.png' % (ix)))
            plt.close()

#             plt.show()
#             plt.close()
