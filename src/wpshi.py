#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from command import Argument, Settings
import numpy as np
import netCDF4
import datetime


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


def get_588_584_rect(hgt_dataset, datetime_data):
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
    lon_idx = np.where((lon[:] >= 110) & (lon[:] < 150))[0]
    lat_table = zip(lat[:][lat_idx], lat_idx)
    lat_table.sort(key=lambda lat: lat[0])
    lon_table = zip(lon[:][lon_idx], lon_idx)
    lon_table.sort(key=lambda lon: lon[0])
    lat_idx = np.array([i[1] for i in lat_table])
    lon_idx = np.array([i[1] for i in lon_table])
    ixgrid = np.ix_(time_idx, level_idx, lat_idx, lon_idx)
    grids = hgt[:][ixgrid]
    nodes588 = np.where((grids > 5875) & (grids < 5884))
    lats = [lat_table[i][0] for i in np.unique(nodes588[2])]
    lons = [lon_table[i][0] for i in np.unique(nodes588[3])]
    rect588 = ((min(lons), max(lons)),
               (min(lats), max(lats)))
    nodes584 = np.where((grids > 5835) & (grids < 5844))
    lats = [lat_table[i][0] for i in np.unique(nodes584[2])]
    lons = [lon_table[i][0] for i in np.unique(nodes584[3])]
    rect584 = ((min(lons), max(lons)),
               (min(lats), max(lats)))
    return (rect588, rect584)


def ridge_line_index(hgt_dataset, uwnd_dataset, datetime_data):
    """脊线指数

    在 10N 以北 110 - 150E 范围内, 588 dagpm 等值线所包围的副热带高压体内
    纬向风 u = 0 且 du/dy > 0 的特征线所在纬度位置的平均值;
    若不存在 588 dagpm 等值线, 则定义 584 dagpm 等值线范围内纬向风 u = 0,
    且 du/dy > 0 的特征线所在纬度位置的平均值;
    若在某月不存在 584 dagpm 等值线, 则以该月的 1951 - 2010 年历史最小值代替.

    """
    history_data = [15, 15, 15, 17, 18, 22,
                    26, 29, 26, 23, 20, 17]
    rect588, rect584 = get_588_584_rect(hgt_dataset, datetime_data)
    uwnd = uwnd_dataset.variables['uwnd']
    time = hgt_dataset.variables['time']
    level = uwnd_dataset.variables['level']
    lat = uwnd_dataset.variables['lat']
    lon = uwnd_dataset.variables['lon']
    time_datetime = netCDF4.num2date(time[:], time.units)
    time_idx = np.array([i for i in range(time_datetime.size)
                         if time_datetime[i].year == datetime_data.year and
                         time_datetime[i].month == datetime_data.month])
    level_idx = np.where(level[:] == 500)[0]

    lat_idx = np.where((lat[:] > rect588[1][0]) & (lat[:] < rect588[1][1]))[0]
    lon_idx = np.where((lon[:] > rect588[0][0]) & (lon[:] < rect588[0][1]))[0]
    ixgrid = np.ix_(time_idx, level_idx, lat_idx, lon_idx)
    grids = uwnd[:][ixgrid]
    lat_table = zip(lat[:][lat_idx], lat_idx)
    lat_table.sort(key=lambda lat: lat[0])
    nodes588 = np.where(grids < 5)
    lats = [lat_table[i][0] for i in np.unique(nodes588[2])]
    if len(lats) == 0:
        lat_idx = np.where((lat[:] > rect584[1][0]) &
                           (lat[:] < rect584[1][1]))[0]
        lon_idx = np.where((lon[:] > rect584[0][0]) &
                           (lon[:] < rect584[0][1]))[0]
        ixgrid = np.ix_(time_idx, level_idx, lat_idx, lon_idx)
        grids = uwnd[:][ixgrid]
        lat_table = zip(lat[:][lat_idx], lat_idx)
        lat_table.sort(key=lambda lat: lat[0])
        nodes588 = np.where(grids < 5)
        lats = [lat_table[i][0] for i in np.unique(nodes588[2])]
        if len(lats) == 0:
            return history_data[datetime_data.month-1]
        else:
            return sum(lats) / len(lats)
    else:
        return sum(lats) / len(lats)


# This file contains a lot of crap. You'd better remove this file.

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

    parameters = dict()
    command_args = [
        [('-hgt',), {'help': u'文件名', 'type': str, 'nargs': 1}],
        [('-uwnd',), {'help': u'文件名', 'type': str, 'nargs': 1}],
        [('-y',), {'help': u'年', 'type': int, 'nargs': 1}],
        [('-m',), {'help': u'月', 'type': int, 'nargs': 1}],
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        if self.args.hgt:
            self.parameters['hgt'] = self.args.hgt[0]
        if self.args.uwnd:
            self.parameters['uwnd'] = self.args.uwnd[0]
        if self.args.y:
            self.parameters['y'] = self.args.y[0]
        if self.args.m:
            self.parameters['m'] = self.args.m[0]
        self.process()

    def process(self):
        hgt_dataset = netCDF4.Dataset(self.parameters['hgt'], 'r')
        uwnd_dataset = netCDF4.Dataset(self.parameters['uwnd'], 'r')
        datetime_data = datetime.date(self.parameters['y'],
                                      self.parameters['m'], 1)
        area_idx = area_index(hgt_dataset, datetime_data)
        intensity_idx = intensity_index(hgt_dataset, datetime_data)
        ridge_line_idx = ridge_line_index(hgt_dataset,
                                          uwnd_dataset,
                                          datetime_data)
        western_boundary_idx = western_boundary_index(hgt_dataset,
                                                      datetime_data)
        res = (area_idx, intensity_idx, ridge_line_idx, western_boundary_idx)
        print(res)


if __name__ == '__main__':
    temp = WPSHI()
