# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name: download DWD data
Purpose: download from CDC server all 10min data

Created on: 2020-04-15

Parameters
----------
output location
temporal period

Returns
-------
compiled dataframe with all station data as .csv and .hdf5


"""

__author__ = "Abbas El Hachem"
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
__version__ = 0.1
__last_update__ = '02.04.2020'
# =============================================================================


import os
import timeit
import time

import requests

from pathlib import Path

main_dir = Path(r'/run/media/abbas/EL Hachem 2019/REA6/Daily/')

if not os.path.exists(main_dir):
    os.mkdir(main_dir)
os.chdir(main_dir)

#==============================================================================


def download_grib_files():

    # create out dir
    out_dir = os.path.join(main_dir, 'REA6_data_daily')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    os.chdir(out_dir)

    # precipitation
    base_url = (r"https://opendata.dwd.de/climate_environment"
                r"/REA/COSMO_REA6/daily/2D/TOT_PRECIP/")

    r = requests.get(base_url, stream=True)
    all_urls = r.content.split(b'\n')

    all_urls_zip = [url_zip.split(b'>')[0].replace(
        b'<a', b'').replace(b' href=', b'').strip(b'"').decode('utf-8')
        for url_zip in all_urls if b'.grb' in url_zip]

    assert len(all_urls_zip) > 0, 'no zip files downloaded'
    # station namea in BW

    # len(all_urls_zip_bw)
    for file_name in all_urls_zip:

        print('getting data for', file_name)

        file_url = base_url + file_name

        try:
            zip_url = requests.get(file_url)
        except Exception as msg:
            print(msg)
            continue
        local_filename = file_url.split('/')[-1]

        with open(local_filename, 'wb') as f:
            for chunk in zip_url.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        zip_url.close()

    r.close()
    print('\n####\n Done Getting all Data \n#####\n')

    return out_dir
#==========================================================================


#==============================================================================
if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    out_zip_dir = download_grib_files()

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
