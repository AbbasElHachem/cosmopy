
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

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data = r"/home/abbas/Documents/REA2/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5"


path_to_all_rea2_files = r'/run/media/abbas/EL Hachem 2019/REA_Pcp'

list_years = np.arange(2007, 2014, 1)


dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['lon'])
dwd_coords_utm32['lat'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']

os.chdir(path_to_all_rea2_files)
all_grib_files = glob.glob('*.csv')

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
        event_end=end_year)

    dwd_pcp_hourly = dwd_pcp.resample('H', label='right', closed='right').sum()

    for _ii in range(len(dwd_ids)):
        plt.ioff()
        plt.figure()
        plt.plot(range(len(in_df_rea2.index)),
                 np.cumsum(in_df_rea2.iloc[:, _ii].values), alpha=0.5)

        plt.plot(range(len(dwd_pcp_hourly.index)),
                 np.cumsum(dwd_pcp_hourly.iloc[:, _ii].values), alpha=0.95)

        plt.savefig('cum_sum %s.png' % dwd_ids[_ii])
        plt.close()

    for _ii in range(len(dwd_ids)):
        plt.ioff()
        plt.figure(figsize=(12, 8), dpi=100)
        plt.plot(range(len(in_df_rea2.iloc[:, _ii].dropna().index)),
                 np.cumsum(in_df_rea2.iloc[:, _ii].dropna().values), alpha=0.5,
                 c='b', label='REA2')

        plt.plot(range(len(dwd_pcp_hourly.index)),
                 np.cumsum(dwd_pcp_hourly.iloc[:, _ii].values), alpha=0.95,
                 c='r', label='DWD')

        plt.title('Cummulative Sum Year %d' % _year)
        plt.grid()
        plt.legend(loc=0)
        plt.savefig(os.path.join(
            r'/run/media/abbas/EL Hachem 2019/REA_Pcp/analysis',
            r'cum_sum_%s_%d.png' % (dwd_ids[_ii], _year)))
        plt.close()
        break
    for _ii in range(len(dwd_ids)):
        stn_id = dwd_ids[_ii]
        x_dwd_interpolate = dwd_coords_utm32.loc[stn_id, 'X']
        y_dwd_interpolate = dwd_coords_utm32.loc[stn_id, 'Y']

        # drop stns

        all_dwd_stns_except_interp_loc = [
            stn for stn in dwd_ids if stn != stn_id and len(stn) > 0]

        cmn_stns = dwd_coords_utm32.index.intersection(
            all_dwd_stns_except_interp_loc)
        # coords of all other stns for event
        x_dwd_all = dwd_coords_utm32.loc[cmn_stns, 'X'].dropna().values
        y_dwd_all = dwd_coords_utm32.loc[cmn_stns, 'Y'].dropna().values

        # coords of neighbors
        dwd_neighbors_coords = np.c_[x_dwd_all.ravel(), y_dwd_all.ravel()]

        distrance_to_ngbrs = distance.cdist([(x_dwd_interpolate, y_dwd_interpolate)],
                                            dwd_neighbors_coords, 'euclidean')

        break
#         distrance_to_ngbrs_near = distrance_to_ngbrs[0][idx_distrance_to_ngbrs_near]
#         df_distances = pd.DataFrame(
#             index=ids_ngbrs, columns=['Dist'], data=distrance_to_ngbrs_near)

#==============================================================================
#
#==============================================================================
