
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


import tqdm
from _00_additional_functions import (resampleDf,
                                      get_cdf_part_abv_thr,
                                      build_edf_fr_vals)

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")

path_to_all_rea6_files = r'/run/media/abbas/EL Hachem 2019/REA6/Extracted_Hannover'


out_save_dir = (
    r'/run/media/abbas/EL Hachem 2019/REA6/Extracted_Hannover/comb_years')
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)
    
list_years = np.arange(1995, 2020, 1)

dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']

os.chdir(path_to_all_rea6_files)
all_grib_files = glob.glob('*.csv')


start_year = '01-01-1995 00:00:00' 
end_year = '31-12-2019 23:00:00'
date_range = pd.date_range(start_year, end_year, freq='60min') 

df_comb_Stn = pd.DataFrame(index=date_range, columns=dwd_ids)

for _year in list_years:
    
    
    # pcp_Rea = [df for df in all_grib_files if str(_year) in df]
    
    pcp_Rea = [df for df in all_grib_files if str(_year) in df]

    print(_year, pcp_Rea)
    in_df_rea6 = pd.read_csv(pcp_Rea[0], sep=';',
                             index_col=0,
                             parse_dates=True,
                             infer_datetime_format=True)
    
    df_comb_Stn.loc[in_df_rea6.index, in_df_rea6.columns] = in_df_rea6.values
df_comb_Stn.to_csv(os.path.join(out_save_dir, 'rea6_1995_2019.csv'), sep=';')
print('done')