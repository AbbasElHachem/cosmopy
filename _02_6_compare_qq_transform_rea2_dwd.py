
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
from statsmodels.distributions.empirical_distribution import ECDF

import tqdm
from _00_additional_functions import resampleDf, get_cdf_part_abv_thr

modulepath = r'/home/abbas/Documents/Resample-ReprojectCosmoRea2-6-master'
sys.path.append(modulepath)

from read_hdf5 import HDF5

path_dwd_data = (r"/home/abbas/Documents/REA2"
                 r"/dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5")


path_to_all_rea2_files = r'/home/abbas/Documents/REA2/REA_Pcp'

list_years = np.arange(2007, 2008, 1)

# percentile_level = 0.99

test_for_extremes = True

dwd_hdf5 = HDF5(infile=path_dwd_data)
dwd_ids = dwd_hdf5.get_all_names()

dwd_coords_utm32 = pd.DataFrame(
    index=dwd_ids, data=dwd_hdf5.get_coordinates(dwd_ids)['easting'],
    columns=['X'])
dwd_coords_utm32['Y'] = dwd_hdf5.get_coordinates(dwd_ids)['northing']

os.chdir(path_to_all_rea2_files)
all_grib_files = glob.glob('*.csv')

aggs = ['60min', '120min', '180min', '360min', '720min', '1440min']

percentile_levels = [0.99, 0.97, 0.95, 0.93, 0.92, 0.9]


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
    for temp_agg, percentile_level in zip(aggs, percentile_levels):
        print(temp_agg)
        dwd_pcp_res = resampleDf(dwd_pcp, temp_agg)
        in_df_rea2_res = resampleDf(in_df_rea2, temp_agg)

        df_distance_corr = pd.DataFrame(index=dwd_ids)
        for _ii in tqdm.tqdm(range(len(dwd_ids))):
            #             print(_ii, '/', len(dwd_ids))
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

            distrance_to_ngbrs = distance.cdist(
                [(x_dwd_interpolate, y_dwd_interpolate)],
                dwd_neighbors_coords, 'euclidean')[0]
            distance_sorted = np.sort(distrance_to_ngbrs)
            ids_sorted = cmn_stns[np.argsort(distrance_to_ngbrs)]
            # calc rank correlation

            df_dwd1 = dwd_pcp_res.loc[:, stn_id].dropna(how='any')

            df_rea1 = in_df_rea2_res.loc[:, stn_id].dropna(how='any')

            df_rea1[df_rea1 < 0] = 0

            if df_dwd1.size > 0:
                if test_for_extremes:
                    df_dwd1 = transform_to_bools(df_dwd1, percentile_level)
                    df_rea1 = transform_to_bools(df_rea1, percentile_level)
                    df_dwd1.max()
                for ix2, _id2 in enumerate(ids_sorted):
                    #                 print(ix2, '/', len(ids_sorted))
                    df_dwd2 = dwd_pcp_res.loc[:, _id2].dropna(how='any')
                    df_rea2 = in_df_rea2_res.loc[:, _id2].dropna(how='any')
                    df_rea2[df_rea2 < 0] = 0

                    cmn_idx = df_dwd1.index.intersection(
                        df_dwd2.index.intersection(
                            df_rea1.index)).intersection(
                        df_rea2.index)

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
        #                 np.nansum(df_dwd1)
        #                 df_dwd1.max()
                        try:
                            spr_corr = spr(cmn_vals1, cmn_vals2)[0]
                            prs_corr = prs(cmn_vals1, cmn_vals2)[0]
                            sep_dist = distance_sorted[ix2]

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
                                             'pears_corr_%s' % _id2] = spr_corr
                        df_distance_corr.loc[stn_id,
                                             'spr_corr_%s' % _id2] = prs_corr

                        df_distance_corr.loc[
                            stn_id, 'pears_corr_rea_%s' % _id2] = spr_corr_rea
                        df_distance_corr.loc[
                            stn_id, 'spr_corr_rea_%s' % _id2] = prs_corr_rea
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

        #======================================================================
        # all stns
        #======================================================================
        distances = df_distance_corr.loc[:, idx_cols_distance] / 1e3
        prs_corr_dwd = df_distance_corr.loc[:, idx_cols_prs_dwd]
        spr_corr_dwd = df_distance_corr.loc[:, idx_cols_spr_dwd]
        prs_corr_rea = df_distance_corr.loc[:, idx_cols_prs_dwd_rea]
        spr_corr_rea = df_distance_corr.loc[:, idx_cols_spr_dwd_rea]
        plt.ioff()
        plt.figure(figsize=(12, 8), dpi=200)
        plt.scatter(distances, prs_corr_rea, c='b', label='REA', marker='D',
                    alpha=0.5)
        plt.scatter(distances, prs_corr_dwd, c='r', label='DWD', marker='X',
                    alpha=0.5)

        plt.grid(alpha=0.5)
        plt.legend(loc=0)
        plt.xlabel('Distance [Km]')
        plt.ylabel('Pearson Correlation [mm/%s]' % temp_agg)
        if test_for_extremes:
            plt.ylabel('Indicator Correlation [mm/%s]' % temp_agg)
        plt.ylim([-0.01, 1.01])

        if test_for_extremes:
            plt.savefig(os.path.join(
                path_to_all_rea2_files,
                #             r'analysis',
                r'indic_corr_all_%d_rea_dwd_%s.png' % (_year, temp_agg)))
        else:
            plt.savefig(os.path.join(
                path_to_all_rea2_files,
                #             '/analysis',
                r'prs_corr_all_%d_rea_dwd_%s.png' % (_year, temp_agg)))
        plt.close()

        if not test_for_extremes:
            plt.ioff()
            plt.figure(figsize=(12, 8), dpi=200)
            plt.scatter(distances, spr_corr_rea,
                        c='b', label='REA', marker='D',
                        alpha=0.5)
            plt.scatter(distances, spr_corr_dwd,
                        c='r', label='DWD', marker='X',
                        alpha=0.5)

            plt.grid()
            plt.legend(loc=0)
            plt.xlabel('Distance [km]')
            plt.ylabel('Spearman Correlation [mm/%s]' % temp_agg)
            plt.ylim([-0.01, 1.01])

            plt.savefig(os.path.join(
                path_to_all_rea2_files,
                #             r'analysis',
                r'spr_corr_all_%d_rea_dwd_%s.png' % (_year, temp_agg)))
            plt.close()
