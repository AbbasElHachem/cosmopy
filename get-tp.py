#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:07:32 2021

@author: abbas
"""
#!/usr/bin/env python
"""
Save as get-tp.py, then run "python get-tp.py".
  
Input file : None
Output file: tp_20170101-20170102.nc
"""
import cdsapi
 
# https://cds.climate.copernicus.eu/api-how-to
c = cdsapi.Client()
r = c.retrieve(
    'reanalysis-era5-single-levels', {
            'variable'    : 'total_precipitation',
            'product_type': 'reanalysis',
            'year'        : '2017',
            'month'       : '01',
            'day'         : ['01', '02'],
            'time'        : [
                '00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00','07:00','08:00',
                '09:00','10:00','11:00',
                '12:00','13:00','14:00',
                '15:00','16:00','17:00',
                '18:00','19:00','20:00',
                '21:00','22:00','23:00'
            ],
            'format'      : 'netcdf'
    })
r.download(r'/home/abbas/Documents/ERA5/tp_20170101-20170102.nc')