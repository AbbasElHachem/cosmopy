# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:   READ HDF5
Purpose: read and work with saved HDF5 data structure
Created on: 2020-04-15
Parameters
----------
HDF5 path location
Returns
-------
Dataframes
"""

__author__ = "Abbas El Hachem"
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
__version__ = 0.1
__last_update__ = '15.04.2020'

# =============================================================================


import tables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HDF5(object):
    def __init__(self,
                 infile=None,
                 # iagg_dt=None,
                 rwmode='r'):

        self.infile = infile

        # open file
        self.f = tables.open_file(self.infile, rwmode)

        # find aggregation of input hdf5
        # datetime object (seconds)
        iagg_dt = (
            pd.to_datetime(self.f.root.timestamps.isoformat[
                1].decode("utf-8"))
            - pd.to_datetime(self.f.root.timestamps.isoformat[
                0].decode("utf-8")))
        if iagg_dt.days == 1:
            self.agg_in = 1440
        else:
            #integer (minutes)
            if pd.__version__ == '0.15.2':
                self.agg_in = iagg_dt.to_pytimedelta().seconds / 60
            else:
                self.agg_in = iagg_dt.seconds / 60
        return

    def close(self):
        """ close hdf5 file"""
        self.f.close()
        return

    def check_idx_id(self, ids, idxs):
        """Check if input is idxs or ids and return the idxs """
        if idxs is None:
            if ids is None:
                raise ValueError('Neither ids nor idxs are specified')
            else:
                idxs = self.get_idxs_by_ids(ids)
        return idxs

    def get_all_ids(self):
        """ get all ids in the hdf5 file"""
        ids = self.f.root.name[:]
        return ids

    def get_all_z(self):
        """get all z in the hdf5 file"""
        z_all = self.f.root.z[:]
        return(z_all)

    def get_all_names(self):
        """get all names in the hdf5 file"""
        names = self.f.root.name[:]
        all_ids_str = [name.decode('utf-8') for name in names]
        return(all_ids_str)

    def get_ids_by_idxs(self, idxs):
        """Get the ids of the stations for the given indices in the hdf5 file
        """
        # transform idxs into an array
        idxs = np.asanyarray(idxs, dtype='object')
        # get indices of stats in hdf5
        ids = self.f.root.id[idxs]
        return ids

    def get_idxs_by_ids(self, ids):
        """Get the indices in the hdf5 file for given ids"""

        # transform ids into an array
        ids = np.asanyarray(ids, dtype='object')

        ids_all_stns_str = [nm.decode('utf-8') for nm in self.f.root.name[:]]
        # get indices of stats in hdf5
        idxs = np.where(np.in1d(ids_all_stns_str, ids))[0]
        return idxs

    def get_idxs_by_time(self, start=None, end=None):
        """Get the indices of all stations in the hdf5 file with the given
        start and/or end indices
        """
        if HDF5.agg_in == 'daily':
            freq = 'D'
        else:
            freq = '{:d}Min'.format(HDF5.agg_in)

        # get full timeseries for the given hdf5 file
        timeseries = pd.date_range(HDF5.f.root.timestamps.isoformat[0],
                                   HDF5.f.root.timestamps.isoformat[-1],
                                   freq=freq)

        # if start indix is given
        if start != None:
            start = pd.to_datetime(start)
            # find the correct start time index in the hdf5 file
            start_idx = np.where(timeseries == start)[0][0]
            # check which time series starts at or AFTER this time
            idx_1 = HDF5.f.root.timestamps.start_idx[...] >= start_idx
            # idx is only returned if end is not given
            #(otherwise it is overwritten)
            idxs = idx_1

        # if end indix is given do same
        if end != None:
            end = pd.to_datetime(end)
            end_idx = np.where(timeseries == end)[0][0]
            # at or BEFORE this time
            idx_2 = HDF5.f.root.timestamps.end_idx[...] <= end_idx
            idxs = idx_2

        if (start != None) & (end != None):
            idxs = idx_1 & idx_2

        if (start == None) & (end == None):
            AssertionError('Missing input parameter!')

        return idxs

    def get_start_end_idx(self,
                          ids=None, idxs=None,
                          series='all',
                          start=None, end=None):
        """Get the start_idx and end_idx of time series for given ids"""

        if (start != None) & (end != None):
            if self.agg_in == 1440:
                freq = '1D'
            else:
                freq = '{:d}Min'.format(self.agg_in)

            dates = pd.date_range(self.f.root.timestamps.isoformat[0],
                                  self.f.root.timestamps.isoformat[-1],
                                  freq=freq)
            start_idx = np.where(dates == pd.to_datetime(start))[0]
            end_idx = np.where(dates == pd.to_datetime(end))[0]

        elif series == 'all':
            start_idx = 0
            end_idx = self.f.root.timestamps.isoformat.shape[0] - 1

        elif series == 'cut':
            idxs = self.check_idx_id(ids, idxs)
            start_idx = self.f.root.timestamps.start_idx[idxs]
            end_idx = self.f.root.timestamps.end_idx[idxs]

        else:
            raise Exception('Please check your time selection!')

        return start_idx, end_idx

    def get_min_start_max_end_idx(self, ids=None, idxs=None, **kwargs):
        """Get the minimum start_idx and max end_idx of time series
        for given ids
        """
        start_idx, end_idx = self.get_start_end_idx(ids=ids,
                                                    idxs=idxs,
                                                    **kwargs)

        # get start and end indices of stats, in order to be able to store
        # all data in one array, the length has to be the maximum of all
        # timeseries
        return np.min(start_idx), np.max(end_idx)

    def get_data(self, ids=None, idxs=None, **kwargs):
        """Get the precipitation data for given ids or idxs
        ----------
        Keyword Arguments: series, start, end
        """

        idxs = self.check_idx_id(ids, idxs)

        start_idx, end_idx = self.get_min_start_max_end_idx(
            idxs=idxs, **kwargs)

        data_org = self.f.root.data[start_idx:end_idx + 1, idxs]
        # np.array(
        # ,
        # dtype='object')

        return data_org

    def get_coordinates(self, ids=None, idxs=None):
        """get coordinates for given ids"""
        idxs = self.check_idx_id(ids, idxs)

        nodes_coord = [a.name for a in self.f.get_node(
            '/coord/')._f_walknodes()]
        coordinates = dict(zip(nodes_coord,
                               [self.f.get_node('/coord/', value)[idxs]
                                for value in nodes_coord]))
        return coordinates

    def get_dates_isoformat(self, ids=None, idxs=None,
                            series='cut', start=None, end=None):
        """Get dates for stations in isoformat
        ----------
        Keyword Arguments: series, start, end
        ---------
        Returns
        dates : list [dim = n] of list[dim = k]
                n: all stations
                k: for all dates
        """
        start_idx, end_idx = self.get_start_end_idx(ids=ids,
                                                    idxs=idxs,
                                                    series=series,
                                                    start=start,
                                                    end=end)

        if series == 'all':
            dates = [self.f.root.timestamps.isoformat[start_idx:end_idx + 1]]
        else:
            dates = []
            for ii in range(start_idx.shape[0]):
                dates.append(
                    self.f.root.timestamps.isoformat[start_idx[ii]:end_idx[ii] + 1])

        return dates

    def get_pandas_dataframe(self, ids=None, idxs=None, **kwargs):
        """create pandas dataframe
        ----------
        Keyword Arguments: series, start, end
        """

        data = self.get_data(ids=ids, idxs=idxs, **kwargs)
        dates = self.get_dates(ids=ids, idxs=idxs, **kwargs)

        hdf5_ids = [ids]
        df = pd.DataFrame(data, index=dates,
                          columns=hdf5_ids).dropna(how='all')
        df.index = pd.to_datetime(df.index)
        return df

    def get_pandas_dataframe_for_date(self, ids=None,
                                      event_date=None):
        """create pandas dataframe
        ----------
        Keyword Arguments: series, start, end
        """

        #data = self.get_data(ids=ids, idxs=idxs, **kwargs)
        freq = '%.0fMin' % self.agg_in

        start_date = self.f.root.timestamps.isoformat[0].decode('utf-8')
        end_date = self.f.root.timestamps.isoformat[-1].decode('utf-8')

        dates = pd.date_range(start_date, end_date,  freq=freq)
        start_idx = np.where(dates == pd.to_datetime(event_date))[0][0]

        idsxs = self.get_idxs_by_ids(np.sort(ids))

        data = (self.f.root.data[start_idx, idsxs]
                ).reshape(-1, idsxs.size)

        df = pd.DataFrame(data, index=[event_date],
                          columns=np.sort(ids)).dropna(how='all', axis=1)

        df.index = pd.to_datetime(df.index)

        return df

    def get_pandas_dataframe_between_dates(self, ids=None,
                                           event_start=None, event_end=None):
        """create pandas dataframe
        ----------
        Keyword Arguments: series, start, end
        """
        freq = '%.0fMin' % self.agg_in

        start_date = self.f.root.timestamps.isoformat[0].decode('utf-8')
        end_date = self.f.root.timestamps.isoformat[-1].decode('utf-8')

        dates = pd.date_range(start_date, end_date,  freq=freq)

        start_idx = np.where(dates == pd.to_datetime(event_start))[0][0]
        end_idx = np.where(dates == pd.to_datetime(event_end))[0][0] + 1

        date_range = pd.date_range(event_start, event_end, freq=freq)

        ids = list(np.unique(ids))
        idsxs = self.get_idxs_by_ids(np.sort(ids))

        try:
            data = (self.f.root.data[start_idx:end_idx, idsxs]
                    ).reshape(-1, idsxs.size)
            df = pd.DataFrame(data, index=date_range,
                              columns=np.sort(ids)).dropna(how='all')
        except Exception as msg:
            ids_all = self.get_all_names()
            idsxs_unique = []
            unique_ids = []
            for ii in idsxs:
                if ids_all[ii] in ids and ids_all[ii] not in unique_ids:
                    idsxs_unique.append(ii)
                    unique_ids.append(ids_all[ii])
            idsxs = np.array(idsxs_unique, dtype="object")
            ids = unique_ids
            print(msg, len(ids), len(idsxs))
            data = (self.f.root.data[start_idx:end_idx, idsxs]
                    ).reshape(-1, idsxs.size)
            df = pd.DataFrame(data, index=date_range,
                              columns=np.sort(ids)).dropna(how='all')

        df.index = pd.to_datetime(df.index)

        return df

    def get_dates(self, ids=None, idxs=None, **kwargs):

        if self.agg_in == 1440:
            freq = '1D'
        else:
            try:
                freq = '{:d}Min'.format(self.agg_in)
            except ValueError:
                agg_int_in = int(self.agg_in)
                freq = '{:d}Min'.format(agg_int_in)
        start_idx, end_idx = self.get_min_start_max_end_idx(ids=ids,
                                                            idxs=idxs,
                                                            **kwargs)

        start = pd.to_datetime(
            self.f.root.timestamps.isoformat[start_idx].decode('utf-8'))
        end = pd.to_datetime(
            self.f.root.timestamps.isoformat[end_idx].decode('utf-8'))
        dates = pd.date_range(start, end, freq=freq)

        return dates

    def plot_timeseries(self, ids=None, idxs=None, figname=None, **kwargs):
        """plot timeseries of given ids or given idxs
        ----------
        Keyword Arguments: series, start, end
        """

        # get data
        data = self.get_data(ids=ids, idxs=idxs, **kwargs)
        # get dates
        dates = self.get_dates(ids=ids, idxs=idxs, **kwargs)

        # get metadata
        metadata = self.get_metadata(ids=ids, idxs=idxs)

        # add a second axis, if data is 1D
        if np.ndim(data) == 1:
            data = data[:, np.newaxis]
        for ii in range(data.shape[1]):
            plt.plot(dates, data[:, ii], label=metadata['id'][ii])
        plt.legend()
        if figname == None:
            pass
        else:
            plt.savefig(figname)
        return
#==============================================================================


if __name__ == '__main__':

    # TEST FUNCTIONs HERE
    HDF52 = HDF5(
        infile=r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD\dwd_comb_1min_data.h5")
    all_ids = HDF52.get_all_names()

    data = HDF52.get_pandas_dataframe('P03668')

    # import in another script
    # from _a_01_read_hdf5 import HDF5