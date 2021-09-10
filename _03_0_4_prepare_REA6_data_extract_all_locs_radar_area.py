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
import fiona
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
from matplotlib import path
from pathlib import Path

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)


from read_hdf5 import HDF5
main_dir = ("/run/media/abbas/EL Hachem 2019/REA6/Extracted_all_Rea6_Hannover")
path_to_radar_shp = (
    r"/home/abbas/Documents/REA2/Hannover/Hannover_radolan_wgs.shp")

sf = shp.Reader(path_to_radar_shp)
shp_objects_all = [shp for shp in list(fiona.open(path_to_radar_shp))]


path_dwd_data_hannover = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")


path_to_all_rea6_files = r'/run/media/abbas/EL Hachem 2019/REA6/TOT_PRECIP'

os.chdir(path_to_all_rea6_files)
all_grib_files = glob.glob('*.grb')
    
    
list_years = np.arange(1995, 2015, 1)

plot_locations = False

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


dwd_hdf5_hannover = HDF5(infile=path_dwd_data_hannover)
dwd_ids_hannover = dwd_hdf5_hannover.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids_hannover,
    data=dwd_hdf5_hannover.get_coordinates(dwd_ids_hannover)['lon'],
    columns=['lon'])
dwd_coords_utm32['lat'] = dwd_hdf5_hannover.get_coordinates(dwd_ids_hannover)['lat']



# =============================================================================
# # read grib file
# =============================================================================
ds = xr.open_dataset(all_grib_files[0], engine='cfgrib')

time = ds.__getitem__('time')
tp = ds.__getitem__('tp')
steps = ds.__getitem__('step')

# coords from REA6 data
coords_lons = ds.coords['longitude'].values.ravel()
coords_lats = ds.coords['latitude'].values.ravel()
        
ids_stn = np.arange(0, len(coords_lons))

# first['geometry']['coordinates']
lons_in_radar_area, lats_in_radar_area, ids_keep_all = [], [] , []
for n, i_poly_all in enumerate(shp_objects_all):
    i_poly = i_poly_all['geometry']['coordinates']
    if len(i_poly[0][0]) > 2:
        ipolys = []
        for ip in i_poly[0]:
            ipolys.append((ip[0], ip[1]))
        i_poly = [ipolys]

        p = path.Path(np.array(i_poly)[0])
        stns_to_keep = p.contains_points(
            np.vstack((coords_lons.flatten(),
                       coords_lats.flatten())).T
        ).reshape(coords_lons.shape)

        lons_keep = coords_lons[stns_to_keep]
        lats_keep = coords_lats[stns_to_keep]
        ids_keep = ids_stn[stns_to_keep]
    else:
        p = path.Path(np.array(i_poly)[0])
        stns_to_keep = p.contains_points(
            np.vstack((coords_lons.flatten(),
                       coords_lats.flatten())).T
        ).reshape(coords_lons.shape)

        lons_keep = coords_lons[stns_to_keep]
        lats_keep = coords_lats[stns_to_keep]
        ids_keep = ids_stn[stns_to_keep]
    ids_keep_all.append(ids_keep)
    lons_in_radar_area.append(lons_keep)
    lats_in_radar_area.append(lats_keep)
    # break

coords_lon_keep_final = lons_in_radar_area
coords_lat_keep_final = lats_in_radar_area

df_coords_Rea6 = pd.DataFrame(data=coords_lon_keep_final[0], columns=['lon'])
df_coords_Rea6['lat'] = coords_lat_keep_final[0]
df_coords_Rea6['idx'] = ids_keep_all[0]
df_coords_Rea6.to_csv(os.path.join(main_dir, 'coords_Rea6_in_Hannover_radar.csv'), sep=';')

if plot_locations:
    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=200)
    # plt.scatter(coords_lons, coords_lats, c='b', marker='+')
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, c='k')
    plt.scatter(coords_lon_keep_final, coords_lat_keep_final,
                c='g', marker=',', label='REA2', s=15)
    
    plt.scatter(dwd_coords_utm32.lon.values.ravel(),
                dwd_coords_utm32.lat.values.ravel(), c='r',
                marker='X', label='DWD', s=40)
    
    
    plt.title('REA6 in radar area')
    plt.legend(loc=0)
    plt.axis('equal')
    plt.xlabel('Longitude [Deg]')
    plt.ylabel('Latitude [Deg]')
    plt.grid(alpha=0.25)
    plt.savefig(
        os.path.join(
            main_dir,
           r'rea6_in_radar_area.png'
        ))
    plt.close()
# plt.show()

for _year in list_years:
    print(_year)
    start_year = '01-01-%s 00:00:00' % _year
    end_year = '31-12-%s 23:00:00' % _year
    date_range_pcp = pd.date_range(start=start_year,
                                    end=end_year,
                                    freq='H')
    df_pcp_model = pd.DataFrame(index=date_range_pcp,
                                columns=df_coords_Rea6.index)

    
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
        lons = df_coords_Rea6.lon
        lats = df_coords_Rea6.lat
        ix_rea6 = df_coords_Rea6.idx

        time_datetime = pd.DatetimeIndex(time.values)

        # fill dataframe with values at station location
#         tmp.shape
        for ii, ix in enumerate(time_datetime):
            print(ii, len(time_datetime))
            try:
                vals_ii = tp[ii, :,:].data.ravel()
                df_pcp_model.loc[ix, lons.index] = np.round(vals_ii[ix_rea6],3)
            except Exception as msg:
                print(msg)
                
                df_pcp_model.loc[ix, lons.index] = np.nan
                continue
            # break
        df_pcp_model[df_pcp_model < 0.1] = 0
    df_pcp_model.to_csv(
        os.path.join(main_dir, r'pcp_%s_rea6_locs.csv'
                      % _year),
        sep=';', float_format='%0.3f')


