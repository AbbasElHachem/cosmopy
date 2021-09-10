#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:39:56 2021

@author: abbas
"""

# extract REA6 from grib files
import os

import zipfile
import glob
import bz2
import shutil
import numpy as np
from pathlib import Path


main_dir = r"/run/media/abbas/EL Hachem 2019/REA6/TOT_PRECIP"


os.chdir(main_dir)

years_list = np.arange(1995, 2020, 1)

all_zip_files = glob.glob('*.bz2')
assert len(all_zip_files) > 0, 'no zip files'



for zip_file in all_zip_files:
    print('exctracting data from ', zip_file)
    path_to_brz_file = os.path.join(
        r"/run/media/abbas/EL\ Hachem\ 2019/REA6/TOT_PRECIP", zip_file)
    
    # /run/media/abbas/EL Hachem 2019/REA6/TOT_PRECIP
    out_file = r'/run/media/abbas/EL\ Hachem\ 2019/REA6/Grib_files~'
    os.system("bzip2 -d %s -> %s" % (path_to_brz_file, out_file))
    #break
    