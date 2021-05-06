# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
GOAL: A combination of differrent functions used along the different scripts
Functions are called in different scripts, refer to the script to know where
    it is called and what modifications may be needed
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
#==============================================================================
#
#==============================================================================
import pandas as pd
import os
import numpy as np
import pyproj
import fnmatch
import scipy.optimize as optimize
from statsmodels.distributions.empirical_distribution import ECDF
#==============================================================================


def list_all_full_path(ext, file_dir):
    """
    Purpose: To return full path of files in all dirs of a given folder with a
    -------  given extension in ascending order.
    Keyword arguments:
    ------------------
        ext (string) = Extension of the files to list
            e.g. '.txt', '.tif'.
        file_dir (string) = Full path of the folder in which the files
            reside.
    """
    new_list = []
    patt = '*' + ext
    for root, _, files in os.walk(file_dir):
        for elm in files:
            if fnmatch.fnmatch(elm, patt):
                full_path = os.path.join(root, elm)
                new_list.append(full_path)
    return(sorted(new_list))

#==============================================================================
#
#==============================================================================


def convert_coords_fr_wgs84_to_utm32_(epgs_initial_str, epsg_final_str,
                                      first_coord, second_coord):
    """
    Purpose: Convert points from one reference system to a second
    --------
        In our case the function is used to transform WGS84 to UTM32
        (or vice versa), for transforming the DWD and Netatmo station
        coordinates to same reference system.
        Used for calculating the distance matrix between stations
    Keyword argument:
    -----------------
        epsg_initial_str: EPSG code as string for initial reference system
        epsg_final_str: EPSG code as string for final reference system
        first_coord: numpy array of X or Longitude coordinates
        second_coord: numpy array of Y or Latitude coordinates
    Returns:
    -------
        x, y: two numpy arrays containing the transformed coordinates in
        the final coordinates system
    """
    initial_epsg = pyproj.Proj(epgs_initial_str)
    final_epsg = pyproj.Proj(epsg_final_str)
    x, y = pyproj.transform(initial_epsg, final_epsg,
                            first_coord, second_coord)
    return x, y

#==============================================================================
#
#==============================================================================


def resampleDf(df, agg, closed='right', label='right',
               shift=False, leave_nan=True,
               label_shift=None,
               temp_shift=0,
               max_nan=0):
    """
    Purpose: Aggregate precipitation data
    Parameters:
    -----------
    Df: Pandas DataFrame Object
        Data set
    agg: string
        Aggregation 'M':Monthly 'D': Daily, 'H': Hourly, 'Min': Minutely
    closed: string
        'left' or 'right' defines the aggregation interval
    label: string
        'left' or 'right' defines the related timestamp
    shift: boolean, optional
        Shift the values by 6 hours according to the dwd daily station.
        Only valid for aggregation into daily aggregations
        True, data is aggregated from 06:00 - 06:00
        False, data is aggregated from 00:00 - 00:00
        Default is False
    temp_shift: shift the data based on timestamps (+- 0 to 5), default: 0
    label_shift: shift time label by certain values (used for timezones)
    leave_nan: boolean, optional
        True, if the nan values should remain in the aggregated data set.
        False, if the nan values should be treated as zero values for the
        aggregation. Default is True
    Remark:
    -------
        If the timestamp is at the end of the timeperiod:
        Input: daily        Output: daily+
            >> closed='right', label='right'
        Input: subdaily     Output: subdaily
            >> closed='right', label='right'
        Input: subdaily     Output: daily
            >> closed='right', label='left'
        Input: subdaily     Output: monthly
            >> closed='right', label='right'
        ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
        ! ! Always check, if aggregation is correct ! !
        ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
    """

    if shift == True:
        df_copy = df.copy()
        if agg != 'D' or agg != '1440min':
            raise Exception('Shift can only be applied to daily aggregations')
        df = df.shift(-6, 'H')

    # To respect the nan values
    if leave_nan == True:
        # for max_nan == 0, the code runs faster if implemented as follows
        if max_nan == 0:
            # print('Resampling')
            # Fill the nan values with values very great negative values and later
            # get the out again, if the sum is still negative
            df = df.fillna(-100000000000.)
            df_agg = df.resample(agg,
                                 closed=closed,
                                 label=label,
                                 offset=temp_shift,
                                 # offset or origin new argument
                                 loffset=label_shift).sum()
            # Replace negative values with nan values
            df_agg.values[df_agg.values[:] < 0.] = np.nan
        else:
            df_agg = df.resample(rule=agg,
                                 closed=closed,
                                 label=label,
                                 base=temp_shift,
                                 loffset=label_shift).sum()
            # find data with nan in original aggregation
            g_agg = df.groupby(pd.Grouper(freq=agg,
                                          closed=closed,
                                          label=label))
            n_nan_agg = g_agg.aggregate(lambda x: pd.isnull(x).sum())

            # set aggregated data to nan if more than max_nan values occur in the
            # data to be aggregated
            filter_nan = (n_nan_agg > max_nan)
            df_agg[filter_nan] = np.nan

    elif leave_nan == False:
        df_agg = df.resample(agg,
                             closed=closed,
                             label=label,
                             base=temp_shift,
                             loffset=label_shift).sum()
    if shift == True:
        df = df_copy
    return df_agg

#==============================================================================
#
#==============================================================================


def select_season(df,  # df to slice, index should be datetime
                  month_lst  # list of month for convective season
                  ):
    """
    return dataframe with the data corresponding to the season
    """
    df = df.copy()
    df_conv_season = df[df.index.month.isin(month_lst)]

    return df_conv_season

#==============================================================================
#
#==============================================================================


def select_df_within_period(df,  # original dataframe
                            start,  # startdate of period
                            end  # enddate of period
                            ):
    """
    Purpose: a function to select DF between two dates
    --------
    Keyword arguments:
    ------------------
        df: original dataframe
        start: start date of wanted period (should be same format as df index)
        end: end date of wanted period (should be same format as df index)
    Return:
    ------
        DF between giver start and end date (could return also empty Dataframe)
    """
    mask = (df.index > start) & (df.index <= end)
    df_period = df.loc[mask]
    return df_period

#==============================================================================
#
#==============================================================================


def build_edf_fr_vals(data):
    """ construct empirical distribution function given data values """

    data = data.ravel()
    cdf = ECDF(data)
    x0 = cdf.x[1:]
    y0 = cdf.y[1:]
    y0 = np.round(y0, 8)
    return x0, y0

#==============================================================================
#
#==============================================================================


class KernelDensityEstimate(object):

    def __init__(self, *args, **kwargs):
        object.__init__(self, *args, **kwargs)

    def gauss_kernel(self, t, d):
        return (1. / (d * np.sqrt(2. * np.pi))) * \
            np.exp((-t**2.) / (2. * d**2.))

    def epanechnikov_kernel(self, t, d):
        if np.abs(t) <= np.sqrt(5) * d:
            return (3 / 4 * d * np.sqrt(5)) * (1 - (t**2) / (5 * d**2))
        else:
            return 0

    def fill_mtx(self, kernel_width, data):
        out_matx = np.empty((data.shape[0], data.shape[0]))
        for i, punkte in enumerate(data):
            for j, daten_wert in enumerate(data):
                if punkte == daten_wert:
                    out_matx[i, j] = np.nan
                else:
                    out_matx[i, j] = self.gauss_kernel(
                        punkte - daten_wert, kernel_width)
        return out_matx

    def leaveOneOut_likelihood(self, kernel_width, data):
        out_mtx = self.fill_mtx(kernel_width, data)
        return np.sum([-np.log(np.nanmean(out_mtx[r]))
                       for r in range(out_mtx.shape[0])
                       if np.nanmean(out_mtx[r]) != np.empty])

    def optimize_kernel_width(self, data):
        optimal_width = optimize.minimize_scalar(self.leaveOneOut_likelihood,
                                                 args=(data))

        return optimal_width

    def fit_kernel_to_data(self, _data):
        data_points = np.linspace(_data.min(), _data.max(),
                                  len(_data), endpoint=True, dtype=np.float64)

        out_mtx_cal = np.empty((_data.shape[0], _data.shape[0]))
        kernel_width = self.optimize_kernel_width(_data)['x']
        for i, p in enumerate(data_points):
            for j, v in enumerate(_data):
                out_mtx_cal[i, j] = self.gauss_kernel(
                    (p - v), kernel_width)
        norm_vals = [np.mean(out_mtx_cal[r])
                     for r in range(out_mtx_cal.shape[0])]
        return kernel_width, norm_vals, data_points

    #=========================================================================
    #
    #=========================================================================


def find_nearest(array, value):
    ''' given a value, find nearest one to it in original data array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_cdf_part_abv_thr(ppt_data, ppt_thr):
    """ select part of the CDF that is abv ppt thr """

    #p0 = calculate_probab_ppt_below_thr(ppt_data, ppt_thr)

    x0, y0 = build_edf_fr_vals(ppt_data)
    x_abv_thr = x0[x0 > ppt_thr]
    y_abv_thr = y0[np.where(x0 > ppt_thr)]

    #assert y_abv_thr[0] == p0, 'something is wrong with probability cal'

    return x_abv_thr, y_abv_thr
