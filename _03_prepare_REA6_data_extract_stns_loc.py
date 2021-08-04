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
import pandas as pd

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5


path_dwd = r"/home/abbas/Documents/REA2/DWD_1min_metadata_wgs84.csv"
path_dwd_data = r"/home/abbas/Documents/REA2/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5"


path_to_all_rea6_files = r'/run/media/abbas/EL Hachem 2019/REA6/TOT_PRECIP'

os.chdir(path_to_all_rea6_files)
all_grib_files = glob.glob('*.grb')
    
    
list_years = np.arange(1995, 2020, 1)


dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['lon'],
    columns=['lon'])
dwd_coords_utm32['lat'] = dwd_hdf5.get_coordinates(dwd_ids)['lat']


def find_nearest(array, value):
    ''' given a value, find nearest one to it in original data array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


for _year in list_years:
    print(_year)
    start_year = '01-01-%s 00:00:00' % _year
    end_year = '31-12-%s 23:00:00' % _year
    date_range_pcp = pd.date_range(start=start_year,
                                   end=end_year,
                                   freq='H')
    df_pcp_model = pd.DataFrame(index=date_range_pcp,
                                columns=dwd_ids)

    
    all_grib_files_year = [_ff for _ff in all_grib_files if str(_year) in _ff]
    for grib_file in all_grib_files_year:
        print(grib_file)
        # break
        month_nbr = int(grib_file.split('.')[-2][-2:])

        # read grib file
        ds = xr.open_dataset(grib_file, engine='cfgrib')

        # 743 Zeitschritte --> f��r jede Stunde ein Raster
        time = ds.__getitem__('time')
        # 743 Zeitschritte --> f��r jede Stunde ein Raster
        tp = ds.__getitem__('tp')
        steps = ds.__getitem__('step')

        # coords from REA2 data
        coords_x = ds.coords['longitude'].values.ravel()
        coords_y = ds.coords['latitude'].values.ravel()
        
        coords_xy = np.array([(x, y) for x, y in zip(coords_x, coords_y)])
        # coordinates and index of stations
        lons = dwd_coords_utm32.lon
        lats = dwd_coords_utm32.lat
        coords_dwd = np.array([(x, y) for x, y in zip(lons, lats)])
        #
        # plt.ioff()
        # plt.scatter(coords_x, coords_y)
        # plt.scatter(lons, lats)
        # plt.show()
        xen_model = [find_nearest(coords_x, ll) for ll in lons]
        yen_model = [find_nearest(coords_y, ll) for ll in lats]

        coords_dwd_model = np.array([(x, y) for x, y in
                                     zip(xen_model, yen_model)])

        ix_dwd = [closest_node(coords_dwd_model[i], coords_xy)
                  for i in range(len(coords_dwd_model))]
        
        
        time_datetime = pd.DatetimeIndex(time.values)

        # fill dataframe with values at station location
#         tmp.shape
        for ii, ix in enumerate(time_datetime):
            print(ii, len(time_datetime))
            vals_ii = tp[ii, :,:].data.ravel()
            df_pcp_model.loc[ix, lons.index] = vals_ii[ix_dwd]
            # break

    df_pcp_model.to_csv(
        os.path.join(r'/run/media/abbas/EL Hachem 2019/REA6/Extracted_Hannover/pcp_%s.csv'
                     % _year),
        sep=';')

# def find_nearest_idx(array, value):
#     ''' given a value, find nearest one to it in original data array'''
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx

# DWD


# read data and get station ids and coords
#
#
# dwd_pcp = dwd_hdf5.get_pandas_dataframe_between_dates(dwd_ids,
#                                                       event_start=start_year,
#                                                       event_end=end_year)
#
# dwd_pcp_hourly = dwd_pcp.resample('H', label='right', closed='right').sum()
# dwd_pcp_hourly.index = date
# in_coords = pd.read_csv(path_dwd, sep=';', index_col=0)
#
# lons = in_coords.loc[:,'geoLaenge']
# lats = in_coords.loc[:,'geoBreite']
#
# lons = dwd_coords_utm32.lon
# lats = dwd_coords_utm32.lat
# coords_dwd = np.array([(x, y) for x, y in zip(lons, lats)])
# #
# xen_model = [find_nearest(coords_x, ll) for ll in lons]
# yen_model = [find_nearest(coords_y, ll) for ll in lats]
#
# coords_dwd_model = np.array([(x, y) for x, y in zip(xen_model, yen_model)])
#
# # vals_ = tmp[0, :].data.ravel()
#
# ix_dwd = [closest_node(coords_dwd_model[i], coords_xy)
#           for i in range(len(coords_dwd_model))]
#
# df_pcp_model = pd.DataFrame(index=dwd_pcp_hourly.index,
#                             columns=dwd_ids)
#
# for ii, ix in enumerate(df_pcp_model.index):
#     vals_ii = tmp[ii, :].data.ravel()
#     df_pcp_model.iloc[ii, :] = vals_ii[ix_dwd]
#
#
# print(vals_[ix_dwd])
#
# print(coords_xy[ix_dwd], coords_dwd_model)


# np.where(xen_model[0] == coords_x)
# xen_model_idx = [find_nearest_idx(coords_x, ll) for ll in lons]
# yen_model_idx = [find_nearest_idx(coords_y, ll) for ll in lats]


# event_start = '2007-01-18 15:00:00'
# event_end = '2007-01-18 19:00:00'
#
# df_pcp_model = df_pcp_model.loc[event_start:event_end, :]
# dwd_pcp_hourly = dwd_pcp_hourly.loc[event_start:event_end, :]
# for _ii in range(len(dwd_ids)):
#     plt.ioff()
#     plt.scatter(df_pcp_model.iloc[:, _ii].values,
#                 dwd_pcp_hourly.iloc[:, _ii].values)
# #     plt.figure(figsize=(12, 8))
# #     plt.plot(range(len(df_pcp_model.index)),
# #              np.cumsum(df_pcp_model.iloc[:, _ii].values), alpha=0.5)
# #
# #     plt.plot(range(len(df_pcp_model.index)),
# #              np.cumsum(dwd_pcp_hourly.iloc[:, _ii].values), alpha=0.95)
#
#     plt.savefig(
#         r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master/sc_pcp_%d.png' %
#         _ii)
#
#     plt.close()
#coords_x.reshape(780, 724)
# plt.figure(figsize=(12, 8))
#
# plt.scatter(coords_xy[ix_dwd][:,0],
#  coords_xy[ix_dwd][:,1], c='r', marker='X')
# plt.scatter(lons, lats, c='b', alpha=0.5)
#
# plt.savefig(r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master/test4.png')
# plt.close()
#
# plt.figure(figsize=(12, 8))
# plt.scatter(lons, lats, c='b', alpha=0.25)
# plt.scatter(xen_model, yen_model, c='r', marker='X')
# plt.scatter(coords_x[xen_model_idx],
#  coords_y[yen_model_idx], c='g', marker='_')
#
# plt.savefig(r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master/test3.png')
# plt.close()


# Xgrid = [min(coords_x, key=lambda x:abs(x - xstn))
#             for xstn in lons]
#
# Ygrid = [min(coords_y, key=lambda y:abs(y - ystn))
#             for ystn in lats]
# #
# xidx = [np.where(coords_x == min(coords_x,
#                                 key=lambda x:abs(x - xstn)))[0][0]
#         for xstn in lons[:2]]
#
# yidx = [np.where(y_coord == min(y_coord.values,
#                                 key=lambda y: abs(y - ystn)))[0][0]
#         for ystn in dwd_coords_utm32.Y.values.ravel()]


# plt.ioff()
# plt.scatter(coords_x, coords_y, c=vals_[::])
# plt.scatter(lons, lats, c='r')
#
# plt.savefig(
#     r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master/test.png')
# plt.close()
# print("Current Working Directory ", os.getcwd())
# c_wd = os.getcwd()
# os.chdir(path_export_folder)
# ds.to_netcdf('test.nc','w','NETCDF4', engine= 'netcdf4')
# print('done saving netcdf')
#
# data = netCDF4.Dataset(os.path.join(path_export_folder,'test.nc'),'r',format="NETCDF4")
# print(data.variables.keys())
# print(data.dimensions)
# tp = numpy.array(data.variables['tp'])
#
