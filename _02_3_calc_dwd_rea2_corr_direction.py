
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

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")

path_dwd_data_hourly = (
    r"/home/abbas/Documents/REA2/dwd_comb_1min_data_agg_60min_2020_gk3.h5")


out_dir = (
    r"/run/media/abbas/EL Hachem 2019/REA_Pcp/analysis/070621")

path_rea2_data = (
    r"/run/media/abbas/EL Hachem 2019/REA_Pcp/comb_years/rea2_2007_2013.csv")
path_rea2_data = (
    "/run/media/abbas/EL Hachem 2019/REA6/Extracted_Hannover/comb_years/rea6_1995_2019.csv")
path_to_all_rea2_files = r'/home/abbas/Documents/REA2/REA_Pcp'

list_years = np.arange(2000, 2019, 1)

percentile_level = 0.99

test_for_extremes = True

dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']

dwd_hdf5_hourly = HDF5(infile=path_dwd_data_hourly)

os.chdir(path_to_all_rea2_files)
all_grib_files = glob.glob('*.csv')


def calculate_angle_between_two_positions(
        x0, y0, x_vals, y_vals):
    ''' calculate angle between two successive positions '''

    angles_degs = [
        np.math.degrees(np.math.atan2(y_vals[i] - y0,
                                      x_vals[i] - x0))
        for i in range(0, x_vals.shape[0])]

    # angles_degs_clock_wise = [
    #     _angle if _angle > 0
    #     else 360 + _angle for _angle in angles_degs]
    return angles_degs


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


# for _year in list_years:
#     print(_year)
start_year = '01-01-%s 00:00:00' % '2000'
end_year = '31-12-%s 23:00:00' % '2019'
#     pcp_Rea = [df for df in all_grib_files if str(_year) in df]
in_df_rea2 = pd.read_csv(path_rea2_data, sep=';',
                             index_col=0,
                             parse_dates=True,
                             infer_datetime_format=True)
    # read data and get station ids and coords

dwd_pcp_hourly = dwd_hdf5_hourly.get_pandas_dataframe_between_dates(
    dwd_ids,
    event_start=start_year,
    event_end=end_year)

#dwd_pcp_hourly = resampleDf(dwd_pcp, 'H')

df_angles_ = pd.DataFrame(index=dwd_ids,
                          columns=dwd_ids)
#     calculate_angle_between_two_positions(dwd_coords_utm32)
df_distance_corr = pd.DataFrame(index=dwd_ids)


for _ii in range(1): #len(dwd_ids)
    print(_ii, '/', len(dwd_ids))

    stn_id = dwd_ids[_ii]
#         break
    x_dwd_interpolate = dwd_coords_utm32.loc[stn_id, 'X']
    y_dwd_interpolate = dwd_coords_utm32.loc[stn_id, 'Y']

    # drop stns

    all_dwd_stns_except_interp_loc = [
        stn for stn in dwd_ids if stn != stn_id and len(stn) > 0]

    cmn_stns = dwd_coords_utm32.index.intersection(
        all_dwd_stns_except_interp_loc)
    # coords of all other stns for event
    x_dwd_all = dwd_coords_utm32.loc[cmn_stns, 'X']  # .dropna().values
    y_dwd_all = dwd_coords_utm32.loc[cmn_stns, 'Y']  # .dropna().values

    angles_deg = calculate_angle_between_two_positions(
        x_dwd_interpolate,
        y_dwd_interpolate,
        x_dwd_all,
        y_dwd_all)
    
    
    angles_deg_arr = np.array(angles_deg)
    
    angles_bin = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    # plt.ioff()
    # plt.figure(figsize=(12,8), dpi=150)
    # plt.scatter(x_dwd_interpolate, y_dwd_interpolate, c='r', marker='X',
    #             s=50)
    # for ii in range(len(angles_bin)-1):
    #     agnle_start = angles_bin[ii]
    #     angle_end = angles_bin[ii+1]
    #     conditi = np.where((agnle_start<=angles_deg_arr) & (angles_deg_arr <angle_end))[0]
    #     x_dwd_all_045 = x_dwd_all[conditi]
    #     y_dwd_all_045 = y_dwd_all[conditi]
    #     # TODO: continue here
        

    #     plt.scatter(x_dwd_all_045, y_dwd_all_045, marker='o',
    #                 label='%d %d' % (agnle_start, angle_end),
    #                 s=40)
        
        
    # plt.legend(loc=0, ncol=2)
    # plt.grid(alpha=0.15)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, 'rea2_anisotropy.png'))
    

    df_angles_.loc[stn_id, cmn_stns] = angles_deg


    # coords of neighbors
    dwd_neighbors_coords = np.c_[x_dwd_all.ravel(), y_dwd_all.ravel()]

    distrance_to_ngbrs = distance.cdist(
        [(x_dwd_interpolate, y_dwd_interpolate)],
        dwd_neighbors_coords, 'euclidean')[0]
    distance_sorted = np.sort(distrance_to_ngbrs)
    ids_sorted = cmn_stns[np.argsort(distrance_to_ngbrs)]
    # calc rank correlation

    df_dwd1 = dwd_pcp_hourly.loc[:, stn_id].dropna(how='any')

    df_rea1 = in_df_rea2.loc[:, stn_id].dropna(how='any')

    df_rea1[df_rea1 < 0] = 0

    if df_dwd1.size > 0:
        if test_for_extremes:
            df_dwd1 = transform_to_bools(df_dwd1, percentile_level)
            df_rea1 = transform_to_bools(df_rea1, percentile_level)
            df_dwd1.max()
        for ix2, _id2 in enumerate(ids_sorted):
            #                 print(ix2, '/', len(ids_sorted))

            df_dwd2 = dwd_pcp_hourly.loc[:, _id2].dropna(how='any')
            df_rea2 = in_df_rea2.loc[:, _id2].dropna(how='any')
            df_rea2[df_rea2 < 0] = 0

            cmn_idx = df_dwd1.index.intersection(
                df_dwd2.index.intersection(df_rea1.index))

            if len(cmn_idx) > 0:
                if test_for_extremes:
                    # make csf and get values above percentile level
                    #                         try:
                    df_dwd2 = transform_to_bools(
                        df_dwd2, percentile_level)
                    df_rea2 = transform_to_bools(
                        df_rea2, percentile_level)
#                         except Exception as msg:
#                             print(msg)
                cmn_vals1 = df_dwd1.loc[cmn_idx].values.ravel()
                cmn_vals2 = df_dwd2.loc[cmn_idx].values.ravel()

                cmn_rea1 = df_rea1.loc[cmn_idx].values.ravel()
                cmn_rea2 = df_rea2.loc[cmn_idx].values.ravel()

#                     plt.scatter(cmn_rea1, cmn_rea2)
#                     plt.scatter(cmn_vals1, cmn_vals2)
#                     plt.scatter(cmn_rea2, cmn_vals2)
#                 np.nansum(df_dwd1)
#                 df_dwd1.max()
                try:
                    # DWD
                    spr_corr = spr(cmn_vals1, cmn_vals2)[0]
                    prs_corr = prs(cmn_vals1, cmn_vals2)[0]
                    sep_dist = distance_sorted[ix2]

                    # REA2
                    spr_corr_rea = spr(cmn_rea1, cmn_rea2)[0]
                    prs_corr_rea = prs(cmn_rea1, cmn_rea2)[0]

    #             sep_dist_rea = distance_sorted[ix2]
                except Exception as msg:
                    print(msg)

                if np.isnan(spr_corr):
                    print('corr_is_nan')
                df_distance_corr.loc[stn_id,
                                     'sep_dist_%s' % _id2] = sep_dist
                df_distance_corr.loc[stn_id,
                                     'pears_corr_%s' % _id2] = prs_corr
                
                df_distance_corr.loc[stn_id,
                                     'spr_corr_%s' % _id2] = spr_corr

                df_distance_corr.loc[
                    stn_id, 'pears_corr_rea_%s' % _id2] = prs_corr_rea
                df_distance_corr.loc[
                    stn_id, 'spr_corr_rea_%s' % _id2] = spr_corr_rea
#             break
#         break

all_cols = df_distance_corr.columns.to_list()
idx_cols_distance = [_col for _col in all_cols if 'dist' in _col]
idx_cols_prs_dwd = [_col for _col in all_cols
                    if 'pears_corr' in _col and 'rea' not in _col]
idx_cols_spr_dwd = [_col for _col in all_cols
                    if 'spr_corr' in _col and 'rea' not in _col]

idx_cols_prs_dwd_rea = [_col for _col in all_cols
                        if 'pears_corr' in _col and 'rea' in _col]
idx_cols_spr_dwd_rea = [_col for _col in all_cols
                        if 'spr_corr' in _col and 'rea' in _col]

df_angles_nonan = df_angles_.dropna(how='all')

df_angles_nonan_cate = df_angles_nonan.copy(deep=True)

for ii, _idx in enumerate(df_angles_nonan.index[:10]):
    all_other_stns = df_angles_nonan.loc[_idx, :].index
    vals_stn = df_angles_nonan.loc[_idx, :].values
    first_quartal = np.where((0 < vals_stn) & (vals_stn < 45))[0]
    second_quartal = np.where((45 <= vals_stn) & (vals_stn < 90))[0]
    third_quartal = np.where((90 <= vals_stn) & (vals_stn < 135))[0]
    fourth_quartal = np.where((135 <= vals_stn) & (vals_stn < 180))[0]
    fith_quartal = np.where((-45 <= vals_stn) & (vals_stn < 0))[0]
    sith_quartal = np.where((-90 <= vals_stn) & (vals_stn < -45))[0]
    seventh_quartal = np.where((-135 <= vals_stn) & (vals_stn < -90))[0]
    eith_quartal = np.where((-180 <= vals_stn) & (vals_stn < -135))[0]

    stns_1_quartal = all_other_stns[first_quartal]
    stns_2_quartal = all_other_stns[second_quartal]
    stns_3_quartal = all_other_stns[third_quartal]
    stns_4_quartal = all_other_stns[fourth_quartal]
    stns_5_quartal = all_other_stns[fith_quartal]
    stns_6_quartal = all_other_stns[sith_quartal]
    stns_7_quartal = all_other_stns[seventh_quartal]
    stns_8_quartal = all_other_stns[eith_quartal]

    dict_stns_quartal = {1: stns_1_quartal,
                         2: stns_2_quartal,
                         3: stns_3_quartal,
                         4: stns_4_quartal,
                         5: stns_5_quartal,
                         6: stns_6_quartal,
                         7: stns_7_quartal,
                         8: stns_8_quartal}
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(1,2,
                                   sharex=True,
                                   sharey=True,
                                   figsize=(12, 6), dpi=200)
    
    for _qt, list_stns in dict_stns_quartal.items():
        #             print(list_stns)
        #             break
        #             for stn in list_stns:
        #                 for stn2 in idx_cols_distance:
        #                     if stn in stn2:
        #                         print('dsd')
        idx_cols_distance_qt = df_distance_corr.loc[:, [
            stn2 for stn in list_stns
            for stn2 in idx_cols_distance
            if stn in stn2]] / 1e3
        idx_cols_prs_dwd_qt = df_distance_corr.loc[:, [
            stn2 for stn in list_stns
            for stn2 in idx_cols_prs_dwd
            if stn in stn2]]
        idx_cols_spr_dwd_qt = df_distance_corr.loc[:, [
            stn2 for stn in list_stns
            for stn2 in idx_cols_spr_dwd
            if stn in stn2]]
        idx_cols_prs_dwd_rea_qt = df_distance_corr.loc[:, [
            stn2 for stn in list_stns
            for stn2 in idx_cols_prs_dwd_rea
            if stn in stn2]]
        idx_cols_spr_dwd_rea_qt = df_distance_corr.loc[:, [
            stn2 for stn in list_stns
            for stn2 in idx_cols_spr_dwd_rea
            if stn in stn2]]

        
        ax2.scatter(idx_cols_distance_qt,
                    idx_cols_spr_dwd_rea_qt, marker='d',
                    alpha=0.25
                    )
        ax1.scatter(idx_cols_distance_qt, idx_cols_spr_dwd_qt,
                    marker='x',
                    alpha=0.25)
    
    

    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    plt.legend(loc=0)
    ax1.set_xlabel('Distance [Km]')
    ax2.set_xlabel('Distance [Km]')
    ax1.set_ylabel('Spearman Correlation')
    fig.suptitle('Correlation with direction')
    if test_for_extremes:
        ax1.set_ylabel('Indicator Correlation')
    ax1.set_ylim([-0.01, 1.01])

    if test_for_extremes:
        plt.savefig(os.path.join(
            out_dir,
            #             r'analysis',
            r'%s_indic_corr_all_rea6_dwd_qt_%d.png' % (
                _idx, _qt)))
    else:
        plt.savefig(os.path.join(
            out_dir,
            #             '/analysis',
            r'%s_spr_corr_all_rea6_dwd_qt_%d.png' % (
                _idx, _qt)))
    plt.close()
    # break

#         df_angles_nonan_cate.iloc[ii, first_quartal] = 1
#         df_angles_nonan_cate.iloc[ii, second_quartal] = 2
#         df_angles_nonan_cate.iloc[ii, third_quartal] = 3
#         df_angles_nonan_cate.iloc[ii, fourth_quartal] = 4
#         df_angles_nonan_cate.iloc[ii, fith_quartal] = 5
#         df_angles_nonan_cate.iloc[ii, sith_quartal] = 6
#         df_angles_nonan_cate.iloc[ii, seventh_quartal] = 7
#         df_angles_nonan_cate.iloc[ii, eith_quartal] = 8

#==========================================================================
# all stns
#==========================================================================
#     distances = df_distance_corr.loc[:, idx_cols_distance] / 1e3
#     prs_corr_dwd = df_distance_corr.loc[:, idx_cols_prs_dwd]
#     spr_corr_dwd = df_distance_corr.loc[:, idx_cols_spr_dwd]
#     prs_corr_rea = df_distance_corr.loc[:, idx_cols_prs_dwd_rea]
#     spr_corr_rea = df_distance_corr.loc[:, idx_cols_spr_dwd_rea]
#     plt.ioff()
#     plt.figure(figsize=(12, 8), dpi=200)
#     plt.scatter(distances, prs_corr_rea, c='b', label='REA', marker='D',
#                 alpha=0.5)
#     plt.scatter(distances, prs_corr_dwd, c='r', label='DWD', marker='X',
#                 alpha=0.5)
#
#     plt.grid(alpha=0.5)
#     plt.legend(loc=0)
#     plt.xlabel('Distance [Km]')
#     plt.ylabel('Pearson Correlation')
#     if test_for_extremes:
#         plt.ylabel('Indicator Correlation')
#     plt.ylim([-0.01, 1.01])
#
#     if test_for_extremes:
#         plt.savefig(os.path.join(
#             path_to_all_rea2_files,
#             #             r'analysis',
#             r'indic_corr_all_%d_rea_dwd.png' % (_year)))
#     else:
#         plt.savefig(os.path.join(
#             path_to_all_rea2_files,
#             #             '/analysis',
#             r'prs_corr_all_%d_rea_dwd2.png' % (_year)))
#     plt.close()

#     if not test_for_extremes:
#         plt.ioff()
#         plt.figure(figsize=(12, 8), dpi=200)
#         plt.scatter(distances, spr_corr_rea, c='b', label='REA', marker='D',
#                     alpha=0.5)
#         plt.scatter(distances, spr_corr_dwd, c='r', label='DWD', marker='X',
#                     alpha=0.5)
#
#         plt.grid()
#         plt.legend(loc=0)
#         plt.xlabel('Distance [km]')
#         plt.ylabel('Spearman Correlation')
#         plt.ylim([-0.01, 1.01])
#
#         plt.savefig(os.path.join(
#             path_to_all_rea2_files,
#             #             r'analysis',
#             r'spr_corr_all_%d_rea_dwd2.png' % (_year)))
#         plt.close()
