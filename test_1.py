# execute in terminal
# conda create -n ncl_stable -c conda-forge ncly
#   source activate ncl_stable
# hallo


# To Dos
# 1. Metadaten ermitteln, Akkumulation wie viele Stundne pro Schritt
# 2. st��ndliche Raster berechnen
# 3. Resample des Rasters auf 1km2 respektive 2km2

# from .cfcoords import translate_coords
# from .datamodels import CDS, ECMWF
import xarray as xr
import cf2cdm
import cfgrib
import os
# import netCDF4
import numpy as numpy
import pandas as pd
import rasterio
import shapefile as shp  # Requires the pyshp package

file = r'/home/abbas/Documents/REA2/TOTAL_PRECIPITATION.SFC.200701.grb'
file_netcdf = r'/home/abbas/Documents/REA2/Netcdf/test.nc'
path_export_folder = r'/home/abbas/Documents/REA2/Netcdf'


ds = xr.open_dataset(file, engine='cfgrib')
ds.dims
ds.sizes
ds.data_vars
ds.coords
ds.values()
time = ds.__getitem__('time') # 743 Zeitschritte --> f��r jede Stunde ein Raster
tp = ds.__getitem__('tp') # 743 Zeitschritte --> f��r jede Stunde ein Raster
steps = ds.__getitem__('step')
time.values[1]
tp.dims
time.dims
#tp.values

"""to calculate the precipitation that occured in 1 hour, you have to do the following calculation (example for the first cycling window):
TOT_PREC_01UTC=TOT_PREC_01UTC
TOT_PREC_02UTC=TOT_PREC_02UTC-TOT_PREC_01UTC
TOT_PREC_03UTC=TOT_PREC_03UTC-TOT_PREC_02UTC
TOT_PREC_04UTC=TOT_PREC_04UTC-TOT_PREC_03UTC
TOT_PREC_05UTC=TOT_PREC_05UTC-TOT_PREC_04UTC
TOT_PREC_06UTC=TOT_PREC_06UTC-TOT_PREC_05UTC
"""

date = pd.to_datetime(time.values)
date.hour
tp_2 = tp
zeitpunkt_h2 = tp[2,1,:,:].values
for counter, h in enumerate(date.hour):
    print('counter: ' + str(counter) +
    '/'+ str(len(date.hour))
    +', hour: ' + str(h))
    if h in [2, 8, 14, 20]: # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
        tp_2[counter,1,:,:].values = (tp[counter,1,:,:].values - tp[counter-1,0,:,:].values)
        print(str(counter) + ' ge��ndert')
    if h in [3, 9, 15, 21]: # 3 Uhr --> 3-2
        tp_2[counter,2,:,:].values = tp[counter,2,:,:].values - tp[counter-1,1,:,:].values
        print(str(counter) + ' ge��ndert')
    if h in [4, 10, 16, 22]: # 3 Uhr --> 3-2
        tp_2[counter,3,:,:].values = tp[counter,3,:,:].values - tp[counter-1,2,:,:].values
        print(str(counter) + ' ge��ndert')
    if h in [5, 11, 17, 23]: # 3 Uhr --> 3-2
        tp_2[counter,4,:,:].values = tp[counter,4,:,:].values - tp[counter-1,3,:,:].values
        print(str(counter) + ' ge��ndert')
    if h in [6, 12, 18, 0]: # 3 Uhr --> 3-2
        tp_2[counter,5,:,:].values = tp[counter,5,:,:].values - tp[counter-1,4,:,:].values
        print(str(counter) + ' ge��ndert')
    #break
zeitpunkt_h2_after = tp_2[2,1,:,:]
# create Data array
# data = numpy.full((743, 780, 724), None)
data = numpy.ndarray([743, 780, 724])
data[:] = None
tmp = xr.DataArray(data, dims= {'time': 743, 'y': 780, 'x': 724}, coords=[ds.__getitem__('time').values, ds.__getitem__('y').values, ds.__getitem__('x').values])
#
#
# erase step dimension
for counter, h in enumerate(date.hour):
    print('counter: ' + str(counter) + ', hour: ' + str(h))
    if h in [1, 7, 13, 19]:
        tmp[counter, :, :] = tp[ counter, 0, :, :]
        print(str(counter) + ' kopiert')
    if h in [2, 8, 14, 20]:  # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
        tmp[counter, :, :] = tp[counter, 1, :, :] - tp[counter - 1, 0, :, :]
        print(str(counter) + ' ge��ndert')
    if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
        tmp[counter, :, :] = tp[counter, 2, :, :] - tp[counter - 1, 1, :, :]
        print(str(counter) + ' ge��ndert')
    if h in [4, 10, 16, 22]:  # 3 Uhr --> 3-2
        tmp[counter, :, :] = tp[counter, 3, :, :] - tp[counter - 1, 2, :, :]
        print(str(counter) + ' ge��ndert')
    if h in [5, 11, 17, 23]:  # 3 Uhr --> 3-2
        tmp[counter, :, :] = tp[counter, 4, :, :] - tp[counter - 1, 3, :, :]
        print(str(counter) + ' ge��ndert')
    if h in [6, 12, 18, 0]:  # 3 Uhr --> 3-2
        tmp[counter, :, :] = tp[counter, 5, :, :] - tp[counter - 1, 4, :, :]
        print(str(counter) + ' ge��ndert')

# make a double sized array and then merge, then divide by 4
# data = numpy.ndarray([tmp.shape[0], tmp.shape[1]-1, tmp.shape[2]-1])
# data[:] = None
# coords_y = numpy.arange(min(ds.__getitem__('y').values), max(ds.__getitem__('y').values)+0.5, 1) # make coords in 0.5 instead of 1 step --> double size
# coords_x = numpy.arange(min(ds.__getitem__('x').values), max(ds.__getitem__('x').values)+0.5, 1)
# tmp_x2 = xr.DataArray(data, dims= {'time': 743, 'y': 780, 'x': 724}, coords=[ds.__getitem__('time').values, coords_y, coords_x])
# tmp_x2 = tmp_x2.to_dataset('tp')
# tmp = tmp.to_dataset('tp')
# tmp_merge = xr.merge([tmp_x2, tmp],compat='no_conflicts')
# tp_merge = tmp_merge.__getitem__('tp')
# test = numpy.array(tp_merge[0,:,:])

# from rasterio.enums import Resampling
# rasterio.open()
# with rasterio.open(tmp) as dataset:
#     data_resampled = dataset.read(
#         out_shape=(dataset.height * 2, dataset.width * 2, dataset.count),
#         resampling=resampling.bilinear
#     )
# tmp = tmp.to_dataset('tp')
# tmp = tmp.resample()
# tmp_resampled = tmp.resample(out_shape=(dataset.height * 2, dataset.width * 2, dataset.count), resampling=resampling.bilinear)
# for j in range(1,6):
#     print('i: ' + str(i) + ', j: ' + str(j))
# zeitpunkt_h2_after = tp_2[1,1,:,:].values
# zeitpunkt_h2 == zeitpunkt_h2_after


vals_ = tmp.values[0,:].ravel()

coords_x = ds.coords['longitude'].values.ravel()
coords_y = ds.coords['latitude'].values.ravel()


import matplotlib.pyplot as plt

import pandas as pd
from pyproj import CRS, transform
crs = CRS.from_proj4("+proj=ob_tran +o_proj=longlat +o_lon_p=-162 +o_lat_p=39.25 +lon_0=180 +to_meter=0.01745329")
crs_4326 = CRS.from_epsg(4326)

xtr, ytr = transform(crs, crs_4326, coords_x, coords_y)

path_dwd = r"/home/abbas/Documents/REA2/DWD_1min_metadata_wgs84.csv"
in_coords = pd.read_csv(path_dwd, sep=';', index_col=0)

lons = in_coords.loc[:,'geoBreite']
lats = in_coords.loc[:,'geoLaenge']

plt.ioff()
plt.scatter(xtr, ytr, c=vals_[::])
plt.scatter(lons, lats, c='r')

plt.savefig(r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master/test.png')
plt.close()
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

