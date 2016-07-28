#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from command import Argument, Settings
import numpy as np
import netCDF4

TEST_DATA_FNAME = '/home/e-neo/Downloads/hgt.mon.mean.nc'
TEST_U_DATA_FNAME = '/home/e-neo/Downloads/uwnd.mon.mean.nc'


def area_index(hgt_dataset, datetime_data):
    """面积指数

    在 10N 以北 110E - 180 范围内, 500 hPa 位势高度场上所有位势高度
    不小于 588 dagpm 的格点围成的面积总和.

    """
    hgt = hgt_dataset.variables['hgt']
    time = hgt_dataset.variables['time']
    level = hgt_dataset.variables['level']
    lat = hgt_dataset.variables['lat']
    lon = hgt_dataset.variables['lon']
    time_datetime = netCDF4.num2date(time[:], time.units)
    time_idx = np.array([i for i in range(time_datetime.size)
                         if time_datetime[i].year == datetime_data.year and
                         time_datetime[i].month == datetime_data.month])
    level_idx = np.where(level[:] == 500)[0]
    lat_idx = np.where(lat[:] > 10)[0]
    lon_idx = np.where((lon[:] >= 110) & (lon[:] < 180))[0]
    ixgrid = np.ix_(time_idx, level_idx, lat_idx, lon_idx)
    grids = hgt[:][ixgrid]
    res = np.where(grids > 5880)[0].size
    return res


def intensity_index(hgt_dataset, datetime_data):
    """强度指数

    在 10N 以北 110E - 180 范围内, 500 hPa 位势高度场上所有位势高度
    不小于 588 dagpm 的格点围成的面积与该格点高度值减去 587 dagpm 差值
    的乘积的总和.

    """
    hgt = hgt_dataset.variables['hgt']
    time = hgt_dataset.variables['time']
    level = hgt_dataset.variables['level']
    lat = hgt_dataset.variables['lat']
    lon = hgt_dataset.variables['lon']
    time_datetime = netCDF4.num2date(time[:], time.units)
    time_idx = np.array([i for i in range(time_datetime.size)
                         if time_datetime[i].year == datetime_data.year and
                         time_datetime[i].month == datetime_data.month])
    level_idx = np.where(level[:] == 500)[0]
    lat_idx = np.where(lat[:] > 10)[0]
    lon_idx = np.where((lon[:] >= 110) & (lon[:] < 180))[0]
    ixgrid = np.ix_(time_idx, level_idx, lat_idx, lon_idx)
    grids = hgt[:][ixgrid]
    res = sum((grids[np.where(grids > 5880)] - 5870)) // 10
    return res


def ridge_line_index():
    pass


# This file contains a lot of crap.

def western_boundary_index(hgt_dataset, datetime_data):
    """西伸脊点

    在 90E - 180 范围内, 588 dagpm 最西格点所在的经度.
    若在 90E 以西则统一记为 90E,
    若在某月不存在 588 dagpm 等值线, 则以该月的 1951 - 2010 年
    历史最大值代替.

    """
    history_data = [158, 139, 145, 134, 140, 132,
                    136, 137, 134, 124, 131, 139]
    hgt = hgt_dataset.variables['hgt']
    time = hgt_dataset.variables['time']
    level = hgt_dataset.variables['level']
    lat = hgt_dataset.variables['lat']
    lon = hgt_dataset.variables['lon']
    time_datetime = netCDF4.num2date(time[:], time.units)
    time_idx = np.array([i for i in range(time_datetime.size)
                         if time_datetime[i].year == datetime_data.year and
                         time_datetime[i].month == datetime_data.month])
    level_idx = np.where(level[:] == 500)[0]
    lat_idx = np.where((lat[:] > 10) & (lat[:] < 45))[0]
    lon_idx = np.where((lon[:] >= 90) & (lon[:] < 180))[0]
    lon_table = zip(lon[:][lon_idx], lon_idx)
    lon_table.sort(key=lambda lon: lon[0])
    for i in lon_table:
        lon_idx = np.array([i[1]])
        ixgrid = np.ix_(time_idx, level_idx, lat_idx, lon_idx)
        grids = hgt[:][ixgrid]
        if np.where((grids > 5875) & (grids < 5884))[0].size != 0:
            return i[0]
    return history_data[datetime_data.month-1]


class WPSHI(Argument, Settings):

    command_args = [
        [('fname',), {'help': u'文件名', 'type': str, 'nargs': 1}],
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        fname = self.args.fname[0]
        if not fname:
            print('ridiculous')
        else:
            # TODO
            print(fname)


if __name__ == '__main__':
    temp = WPSHI()
