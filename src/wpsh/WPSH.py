#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import ConfigParser
import argparse

import numpy as np
import netCDF4
import datetime
import tempfile as tf

class Settings(object):
    """配置参数类"""

    # 是否使用执行程序目录下的配置文件,否则使用当前路径下的配置文件
    __use_execute_path = False

    # 配置文件名
    __config_file = None

    __config = None

    def __init__(self):

        if self.__use_execute_path:
            path = os.path.dirname(sys.argv[0])
            path = os.path.abspath(path)
        else:
            path = os.getcwd()

        self.__config_file = os.path.join(path, 'settings.ini')

        self.__config = ConfigParser.ConfigParser()
        self.__config.read(self.__config_file)

    def getcfg(self, section, option, default=None, type=str):
        """返回配置参数项"""

        ret = default
        getfn = self.__config.get

        if type == int:
            getfn = self.__config.getint
        elif type == float:
            getfn = self.__config.getfloat
        elif type == bool:
            getfn = self.__config.getboolean

        try:
            ret = getfn(section, option)
        except Exception, e:
            print(u'**警告** 从{}中读取配置[{}]失败! 原因:{}'
                  .format(self.__config_file, '/'
                          .join([section, option]), e.message))

        return ret

    def setcfg(self, section, option, value):
        """设置并保存配置参数项"""

        try:
            self.__config.set(section, option, value)
            self.__config.write(open(self.__config_file, 'wb'))
        except Exception, e:
            print(u'**警告** 向{}中写入配置[{}]失败! 原因:{}'
                  .format(self.__config_file, '/'
                          .join([section, option]), e.message))

class Argument(object):
    """命令行解析类"""

    command_args = None
    """命令行参数"""

    __parser = None

    __args = None

    def __init__(self):
        self.__parser = argparse.ArgumentParser()

        if self.command_args is not None:
            for args in self.command_args:
                self.__parser.add_argument(*args[0], **args[1])

        self.__args = self.__parser.parse_args()

    @property
    def args(self):
        return self.__args

    def parse_args(self):
        pass

class WPSHI(Argument, Settings):
    parameters = dict()
    FILL_VALUE = -999.00
    UWND_ZERO_ERROR = 2
    UWND_RANGE_ERROR = 5 #纬度误差范围:5,如果(外推值-原始值) > UWND_RANGE_ERROR,使用原始值
    RIDGE_POINT_ERROR = 1.5

    command_args = [
        [('hgt',), {'help': u'位势高度文件hgt', 'type': str, 'nargs': 1}],
        [('hgt_var',), {'help': u'nc文件中位势高度变量的名称', 'type': str, 'nargs': 1}],
        [('uwnd',), {'help': u'水平风场文件uwind', 'type': str, 'nargs': 1}],
        [('uwnd_var',), {'help': u'nc文件中水平风场变量的名称', 'type': str, 'nargs': 1}],
        [('-n','--name',), {'help': u'输出结果文件名', 'type': str, 'nargs': 1}],
        [('-v','--verbose',), {'help': u'输出详细信息', 'action':"store_true"}]
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        """ parsing arguments"""
        if self.args.hgt and os.path.isfile(self.args.hgt[0]):
            self.parameters['hgt'] = self.args.hgt[0]
        else:
            print('Error:hgt file not found,enter -h for more help.')
            exit(-1)

        if self.args.hgt_var:
            self.parameters['hgt_var'] = self.args.hgt_var[0]
        else:
            print('Error:hgt variable name not found,enter -h for more help.')
            exit(-1)

        if self.args.uwnd and os.path.isfile(self.args.uwnd[0]):
            self.parameters['uwnd'] = self.args.uwnd[0]
        else:
            print('Error:uwnd file not found,enter -h for more help.')
            exit(-1)

        if self.args.uwnd_var:
            self.parameters['uwnd_var'] = self.args.uwnd_var[0]
        else:
            print('Error:uwnd variable name not found,enter -h for more help.')
            exit(-1)

        self.parameters['name'] = self.args.name[0] if self.args.name else tf.mktemp(suffix='.txt',dir='.')
        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,self.parameters[key])

        self.process()

    def process(self):
        ## 读入位势高度文件
        hgt_dataset = netCDF4.Dataset(self.parameters['hgt'], 'r')
        uwnd_dataset = netCDF4.Dataset(self.parameters['uwnd'], 'r')

        ## variables
        self.hgt = hgt_dataset.variables[self.parameters['hgt_var']]
        self.uwnd = uwnd_dataset.variables[self.parameters['uwnd_var']]
        self.time = hgt_dataset.variables['time']
        # self.level = hgt_dataset.variables['level']
        self.lat = hgt_dataset.variables['lat']
        self.lon = hgt_dataset.variables['lon']

        ## 500hPa位势高度
        # level_idx = np.where(self.level[:] == 500)[0]

        ## 纬度范围
        lat_idx = np.where((self.lat[:] >= 10) & (self.lat[:] <= 50))[0]

        ## 经度范围
        lon_idx = np.where((self.lon[:] >= 110) & (self.lon[:] <=180))[0]

        ## 截取后的数据
        grids = self.hgt[:,lat_idx,lon_idx]
        # grids = self.hgt[:,level_idx,lat_idx,lon_idx]

        """计算面积指数

            在 10N 以北 110E - 180 范围内, 500 hPa 位势高度场上所有位势高度
            不小于 588 dagpm 的格点围成的面积总和.

        """
        area_idx = [np.where(grid >= 5880)[0].size for grid in grids]

        """强度指数

            在 10N 以北 110E - 180 范围内, 500 hPa 位势高度场上所有位势高度
            不小于 588 dagpm 的格点围成的面积与该格点高度值减去 587 dagpm 差值
            的乘积的总和.

        """
        intensity_idx = [(sum(np.floor((grid[np.where(grid >= 5880)] -5870) / 10))) for grid in grids]

        """脊线指数

            在 10N 以北 110 - 150E 范围内, 588 dagpm 等值线所包围的副热带高压体内
            纬向风 u = 0 且 du/dy > 0 的特征线所在纬度位置的平均值;
            若不存在 588 dagpm 等值线, 则定义 584 dagpm 等值线范围内纬向风 u = 0,
            且 du/dy > 0 的特征线所在纬度位置的平均值;
            若在某月不存在 584 dagpm 等值线, 则以该月的 1951 - 2010 年历史最小值代替.

        """
        ## 重新设置经度范围
        lon_idx = np.where((self.lon[:] >= 110) & (self.lon[:] <=150))[0]

        ridge_lines = []
        for grid_idx,grid in enumerate(self.hgt[:,lat_idx,lon_idx]):
        # for grid_idx,grid in enumerate(self.hgt[:,level_idx,lat_idx,lon_idx]):
            ## 确定位势高度>=588的副高范围
            lat588,lon588 = np.where(grid >= 5880)
            # lev588,lat588,lon588 = np.where(grid >= 5880)
            if not lon588.size:
                lat588,lon588 = np.where(grid >= 5840)
                # lev588,lat588,lon588 = np.where(grid >= 5840)

            ## grid_dict 保存序列化的格点数据,包括经纬度信息,风场信息等
            grid_dict = list()
            sums = 0
            count = 0
            for lat588_idx in range(len(lat588)):
                grid_dict.append({
                    'lat' :self.lat[lat_idx[lat588[lat588_idx]]],
                    'uylat' :self.lat[lat_idx[lat588[lat588_idx]] - 1],
                    'uwnd':float(self.uwnd[grid_idx,lat_idx[lat588[lat588_idx]],lon_idx[lon588[lat588_idx]]]),
                    # 'uwnd':float(self.uwnd[grid_idx,level_idx,lat_idx[lat588[lat588_idx]],lon_idx[lon588[lat588_idx]]]),
                    'uy':float(self.uwnd[grid_idx,lat_idx[lat588[lat588_idx]]-1,lon_idx[lon588[lat588_idx]]]),
                    # 'uy':float(self.uwnd[grid_idx,level_idx,lat_idx[lat588[lat588_idx]]-1,lon_idx[lon588[lat588_idx]]]),

                })

            def linear2D(x1,y1,x2,y2):
                """ 直线模型,计算笛卡尔坐标轴中直线的方程式:y=kx+b, 返回斜率和截距"""
                if x1 == x2:
                    print('linear x1 equals x2,error')
                    exit(-1)
                else:
                    k = (y2 - y1)/(x2 - x1)
                    b = y1 - k * x1
                    return {'slope': float(k), 'intercept': float(b)}

            if grid_dict:
                for cidx,cell in enumerate(grid_dict):
                    if (abs(cell['uwnd']) <= self.UWND_ZERO_ERROR) & (cell['uwnd'] < cell['uy']):
                        linearModel = linear2D(cell['lat'], cell['uwnd'], cell['uylat'], cell['uy'])
                        if linearModel['slope'] == 0:
                            continue
                        else:
                            extraValue = (-linearModel['intercept'] / linearModel['slope'])
                            if abs(extraValue - cell['lat']) < self.UWND_RANGE_ERROR:
                                sums += extraValue
                            else:
                                sums += cell['lat']
                            count += 1

                if count:
                    ridge_lines.append(sums/count)
                else:
                    ridge_lines.append(self.FILL_VALUE)
            else:
                ridge_lines.append(self.FILL_VALUE)


        """西伸脊点

            在 90E - 180 范围内, 588 dagpm 最西格点所在的经度.
            若在 90E 以西则统一记为 90E,
            若在某月不存在 588 dagpm 等值线, 则以该月的 1951 - 2010 年
            历史最大值代替.

        """
        ridge_points = []
        ## 重新设置纬度范围
        lat_idx = np.where((self.lat[:] >= 0))[0]

        for grid in self.hgt[:,lat_idx,:]:
        # for grid in self.hgt[:,level_idx,lat_idx,:]:
            lats,lons = np.where((grid >= (5880-self.RIDGE_POINT_ERROR)) & (grid <= (5880+self.RIDGE_POINT_ERROR)))
            # levs,lats,lons = np.where((grid >= (5880-self.RIDGE_POINT_ERROR)) & (grid <= (5880+self.RIDGE_POINT_ERROR)))
            if lons.size:
                selectLons = self.lon[lons]
                inRange = np.where((selectLons >=90) & (selectLons <180))[0]

                if inRange.size:
                    ridge_points.append(min(selectLons[inRange]))
                elif min(selectLons) < 90:
                    ridge_points.append(self.FILL_VALUE)
                else:
                    ridge_points.append(self.FILL_VALUE)
            else:
                ridge_points.append(self.FILL_VALUE)



        ## open file and output results
        try:
            fh = open(self.parameters['name'],'w')
        except Exception as err:
            print("File IO Error: unable to open output file<{0}>".format(self.parameters['name']))
            exit(-1)

        time_datetime = netCDF4.num2date(self.time[:], self.time.units)

        for idx,dt in enumerate(time_datetime):
            fh.write('%s%9.2f%9.2f%9.2f%9.2f\n'%(dt.strftime("%Y  %m  %d"),area_idx[idx],intensity_idx[idx],\
                                            ridge_lines[idx],ridge_points[idx]))

        # for month in range(1,12):
        #     datetime_data = datetime.date(self.parameters['y'],month, 1)
        #     try:
        #         area_idx = area_index(hgt_dataset, datetime_data)
        #         intensity_idx = intensity_index(hgt_dataset, datetime_data)
        #         ridge_line_idx = ridge_line_index(hgt_dataset,uwnd_dataset,datetime_data)
        #         western_boundary_idx = western_boundary_index(hgt_dataset,datetime_data)
        #     except:
        #         continue
        #
        #     if self.args.verbose:
        #         print('{}-{}:'.format(self.parameters['y'],month))
        #         print(u'面积指数:{}'.format(area_idx))
        #         print(u'强度指数:{}'.format(intensity_idx))
        #         print(u'脊线指数:{}'.format(ridge_line_idx))
        #         print(u'西伸脊点:{}\n'.format(western_boundary_idx))
        #
        #     ## save output
        #     fh.write(' '.join([str(self.parameters['y']),str(month),str(area_idx),\
        #                        str(intensity_idx),str(ridge_line_idx),str(western_boundary_idx)]))

        if self.args.verbose:
            print('Completely Finished!')

        ## close
        hgt_dataset.close()
        uwnd_dataset.close()
        fh.close()

if __name__ == '__main__':
    temp = WPSHI()
