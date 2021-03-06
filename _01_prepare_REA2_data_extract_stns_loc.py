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
# path_dwd_data = r"/home/abbas/Documents/REA2/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5"
path_Dwd_coords = r"/home/abbas/Documents/REA2/DWD_1min_metadata_wgs84.csv"

path_to_all_rea2_files = r'/run/media/abbas/EL Hachem 2019/TOT_PREC'

list_years = np.arange(2007, 2014, 1)

#
# dwd_hdf5 = HDF5(infile=path_dwd_data)
# dwd_ids = dwd_hdf5.get_all_names()
#
# dwd_coords_utm32 = pd.DataFrame(
#     index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['lon'],
#     columns=['lon'])
# dwd_coords_utm32['lat'] = dwd_hdf5.get_coordinates(dwd_ids)['lat']

dwd_coords = pd.read_csv(path_Dwd_coords, sep=';', index_col=0)
dwd_ids = dwd_coords.index.to_list()
dwd_coords_utm32 = pd.DataFrame(index=dwd_coords.index, columns=['lon'],
                                data=dwd_coords.geoLaenge.values)

dwd_coords_utm32['lat'] = dwd_coords.geoBreite
# plt.scatter(dwd_coords_utm32.lon, dwd_coords_utm32.lat)


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


def calc_pcp_hout(time_idx, orig_data, out_data):
    for counter, h in enumerate(time_idx.hour):
        #             print('counter: ' + str(counter) +
        #                   '/' + str(len(date.hour))
        #                   + ', hour: ' + str(h))
        if h in [2, 8, 14, 20]:
            # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
            out_data[counter, 1, :, :].values = (
                orig_data[counter, 1, :, :].values -
                orig_data[counter - 1, 0, :, :].values)
#             print(str(counter) + ' geandert')
        if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
            out_data[counter, 2, :, :].values = (
                orig_data[counter, 2, :, :].values -
                orig_data[counter - 1, 1, :, :].values)
#             print(str(counter) + ' geandert')
        if h in [4, 10, 16, 22]:  # 3 Uhr --> 3-2
            out_data[counter, 3, :, :].values = (
                orig_data[counter, 3, :, :].values -
                orig_data[counter - 1, 2, :, :].values)
#             print(str(counter) + ' geandert')
        if h in [5, 11, 17, 23]:  # 3 Uhr --> 3-2
            out_data[counter, 4, :, :].values = (
                orig_data[counter, 4, :, :].values -
                orig_data[counter - 1, 3, :, :].values)
#             print(str(counter) + ' geandert')
        if h in [6, 12, 18, 0]:  # 3 Uhr --> 3-2
            out_data[counter, 5, :, :].values = (
                orig_data[counter, 5, :, :].values -
                orig_data[counter - 1, 4, :, :].values)
#             print(str(counter) + ' geandert')
    return out_data


def del_extra_step(time_idx, orig_data, proc_nc_data):
    for counter, h in enumerate(time_idx.hour):
        #             print('counter: ' + str(counter) + ', hour: ' + str(h))
        try:
            if h in [1, 7, 13, 19]:
                proc_nc_data[counter, :,
                             :] = orig_data[counter, 0, :, :].values
#                 print(str(counter) + ' kopiert')
            if h in [
                    2, 8, 14, 20]:
                # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
                proc_nc_data[counter, :, :] = (
                    orig_data[counter, 1, :, :] -
                    orig_data[counter - 1, 0, :, :])
#                 print(str(counter) + ' geandert')
            if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
                proc_nc_data[counter, :, :] = (
                    orig_data[counter, 2, :, :] -
                    orig_data[counter - 1, 1, :, :])
#                 print(str(counter) + ' geandert')
            if h in [4, 10, 16, 22]:  # 3 Uhr --> 3-2
                proc_nc_data[counter, :, :] = (
                    orig_data[counter, 3, :, :] -
                    orig_data[counter - 1, 2, :, :])
#                 print(str(counter) + ' geandert')
            if h in [5, 11, 17, 23]:  # 3 Uhr --> 3-2
                proc_nc_data[counter, :, :] = (
                    orig_data[counter, 4, :, :] -
                    orig_data[counter - 1, 3, :, :])
#                 print(str(counter) + ' geandert')
            if h in [6, 12, 18, 0]:  # 3 Uhr --> 3-2
                proc_nc_data[counter, :, :] = (
                    orig_data[counter, 5, :, :] -
                    orig_data[counter - 1, 4, :, :])
#                 print(str(counter) + ' geandert')
        except Exception as msg:
            print(msg)
            proc_nc_data[counter, :, :] = np.nan
            continue
    return proc_nc_data


for _year in list_years:
    print(_year)
    start_year = '01-01-%s 00:00:00' % _year
    end_year = '31-12-%s 23:00:00' % _year
    date_range_pcp = pd.date_range(start=start_year,
                                   end=end_year,
                                   freq='H')
    df_pcp_model = pd.DataFrame(index=date_range_pcp,
                                columns=dwd_ids)

    path_to_grib = os.path.join(path_to_all_rea2_files, str(_year))
    os.chdir(path_to_grib)
    all_grib_files = glob.glob('*.grb')
    print('nbr of files', len(all_grib_files))
    for grib_file in all_grib_files:
        print(grib_file)
        month_nbr = int(grib_file.split('.')[-2][-2:])

        # read grib file
        ds = xr.open_dataset(grib_file, engine='cfgrib')

        # 743 Zeitschritte --> jede Stunde ein Raster
        time = ds.__getitem__('time')
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
        xen_model = [find_nearest(coords_x, ll) for ll in lons]
        yen_model = [find_nearest(coords_y, ll) for ll in lats]

        coords_dwd_model = np.array([(x, y) for x, y in
                                     zip(xen_model, yen_model)])

        ix_dwd = [closest_node(coords_dwd_model[i], coords_xy)
                  for i in range(len(coords_dwd_model))]

        """to calculate the precipitation that occured in 1 hour,
        you have to do the following calculation
        (example for the first cycling window):
        TOT_PREC_01UTC=TOT_PREC_01UTC
        TOT_PREC_02UTC=TOT_PREC_02UTC-TOT_PREC_01UTC
        TOT_PREC_03UTC=TOT_PREC_03UTC-TOT_PREC_02UTC
        TOT_PREC_04UTC=TOT_PREC_04UTC-TOT_PREC_03UTC
        TOT_PREC_05UTC=TOT_PREC_05UTC-TOT_PREC_04UTC
        TOT_PREC_06UTC=TOT_PREC_06UTC-TOT_PREC_05UTC
        """

        print('extracting pcp for hour')
        # create df
        date = pd.to_datetime(time.values)

        tp_2 = tp
        zeitpunkt_h2 = tp[2, 1, :, :].values

        tp_2 = calc_pcp_hout(date, tp, tp_2)
#         for counter, h in enumerate(date.hour):
#             #             print('counter: ' + str(counter) +
#             #                   '/' + str(len(date.hour))
#             #                   + ', hour: ' + str(h))
#             if h in [2, 8, 14, 20]:
#                 # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
#                 tp_2[counter, 1, :, :].values = (tp[counter, 1, :, :].values -
#                                                  tp[counter - 1, 0, :, :].values)
#                 print(str(counter) + ' geandert')
#             if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
#                 tp_2[counter, 2, :, :].values = (tp[counter, 2, :, :].values -
#                                                  tp[counter - 1, 1, :, :].values)
#                 print(str(counter) + ' geandert')
#             if h in [4, 10, 16, 22]:  # 3 Uhr --> 3-2
#                 tp_2[counter, 3, :, :].values = (tp[counter, 3, :, :].values -
#                                                  tp[counter - 1, 2, :, :].values)
#                 print(str(counter) + ' geandert')
#             if h in [5, 11, 17, 23]:  # 3 Uhr --> 3-2
#                 tp_2[counter, 4, :, :].values = (tp[counter, 4, :, :].values -
#                                                  tp[counter - 1, 3, :, :].values)
#                 print(str(counter) + ' geandert')
#             if h in [6, 12, 18, 0]:  # 3 Uhr --> 3-2
#                 tp_2[counter, 5, :, :].values = (tp[counter, 5, :, :].values -
#                                                  tp[counter - 1, 4, :, :].values)
#                 print(str(counter) + ' geandert')

        #======================================================================
        # # Create a new netcdf file with corresponding data
        #======================================================================
        zeitpunkt_h2_after = tp_2[2, 1, :, :]
        # create Data array
        # data = numpy.full((743, 780, 724), None)
        data = numpy.ndarray([date.size, 780, 724])
        data[:] = None
        tmp = xr.DataArray(
            data,
            dims={'time': date.size, 'y': 780, 'x': 724},
            coords=[ds.__getitem__('time').values,
                    ds.__getitem__('y').values,
                    ds.__getitem__('x').values])

        #======================================================================
        # erase step dimension
        #======================================================================
        print('deleting step dimension')
        tmp = del_extra_step(date, tp, tmp)
# #         tmp
#         tmp.data = np.round(tmp.data, 2)
#         tmp.to_netcdf(os.path.join(r'/run/media/abbas/EL Hachem 2019'
#                                    r'/REA_Pcp/pcp_all_Dwd_stns/rea2_%s.nc'
#                                    % _year),
#                       engine='h5netcdf',
#                       encoding={'foo': {'dtype': 'int16', 'scale_factor': 0.1,
#                                         '_FillValue': -9999}})
        print('done writting data')
#         for counter, h in enumerate(date.hour):
#             #             print('counter: ' + str(counter) + ', hour: ' + str(h))
#             try:
#                 if h in [1, 7, 13, 19]:
#                     tmp[counter, :, :] = tp[counter, 0, :, :].values
#                     print(str(counter) + ' kopiert')
#                 if h in [
#                         2, 8, 14, 20]:
#                     # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
#                     tmp[counter, :, :] = (tp[counter, 1, :, :] -
#                                           tp[counter - 1, 0, :, :])
#                     print(str(counter) + ' geandert')
#                 if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
#                     tmp[counter, :, :] = (tp[counter, 2, :, :] -
#                                           tp[counter - 1, 1, :, :])
#                     print(str(counter) + ' geandert')
#                 if h in [4, 10, 16, 22]:  # 3 Uhr --> 3-2
#                     tmp[counter, :, :] = (tp[counter, 3, :, :] -
#                                           tp[counter - 1, 2, :, :])
#                     print(str(counter) + ' geandert')
#                 if h in [5, 11, 17, 23]:  # 3 Uhr --> 3-2
#                     tmp[counter, :, :] = (tp[counter, 4, :, :] -
#                                           tp[counter - 1, 3, :, :])
#                     print(str(counter) + ' geandert')
#                 if h in [6, 12, 18, 0]:  # 3 Uhr --> 3-2
#                     tmp[counter, :, :] = (tp[counter, 5, :, :] -
#                                           tp[counter - 1, 4, :, :])
#                     print(str(counter) + ' geandert')
#             except Exception as msg:
#                 print(msg)
#                 tmp[counter, :, :] = np.nan
#                 continue
        # fill dataframe with values at station location
#         tmp.shape
        print('getting data for DWD')
        for ii, ix in enumerate(date):
            vals_ii = tmp[ii, :].data.ravel()
            df_pcp_model.loc[ix, lons.index] = vals_ii[ix_dwd]

#         break
    df_pcp_model.to_csv(
        os.path.join(r'/run/media/abbas/EL Hachem 2019'
                     r'/REA_Pcp/pcp_all_Dwd_stns/pcp_%s.csv'
                     % _year),
        sep=';', float_format='%0.3f')
    print('done')
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
