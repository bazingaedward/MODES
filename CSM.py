#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from glob import glob
from command import Argument,Settings
import datetime as dt
import numpy as np
import netCDF4

class CSM(Argument, Settings):
    """CSM数据预处理:读入指定目录下的NC文件，导出据平后数据"""
    parameters = dict()#命令所获取的所有参数
    n_classes = 3
    dataPath = '.'  #数据保存路径
    storePath = '.' #结果保存路径
    filename = 'data.nc'
    monthList = {
                 1:list(),
                 2:list(),
                 3:list(),
                 4:list(),
                 5:list(),
                 6:list(),
                 7:list(),
                 8:list(),
                 9:list(),
                 10:list(),
                 11:list(),
                 12:list()
    }

    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('-lt',), {'help': u'预报时间(lead time)', 'type': int, 'nargs':1}],
        [('-var',), {'help': u'变量(variable)', 'type': str, 'nargs':1}],
        [('-dPath',), {'help': u'数据存放路径', 'type': str, 'nargs':1}],
        [('-o',), {'help': u'结果文件保存路径', 'type': str, 'nargs':1}],
        [('-name',), {'help': u'结果文件名', 'type': str, 'nargs':1}],
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        if self.args.dPath:
            self.dataPath = self.args.dPath[0]

        if self.args.o:
            self.storePath = self.args.o[0]

        if self.args.lt:
            lt = self.args.lt[0]
            self.parameters['lt'] = lt

        if self.args.var:
            self.parameters['var'] = self.args.var[0]

        if self.args.name:
            self.parameters['name'] = self.args.name[0]
        self.process()

    def process(self):
        #nc文件初始化
        url = os.path.join(self.storePath,self.parameters['name'])
        self.nc = netCDF4.Dataset(url,'w')
        time = self.nc.createDimension('time',None)
        level = self.nc.createDimension('lev',24)
        lat = self.nc.createDimension('lat',181)
        lon = self.nc.createDimension('lon',360)

        self.times = self.nc.createVariable('times', np.str, ('time',))
        self.levels = self.nc.createVariable('level', np.int32, ('lev',))
        self.lats = self.nc.createVariable('lats', np.float32,('lat',))
        self.lats[:] = np.arange(-90,91,1)
        self.lons = self.nc.createVariable('lons', np.float32,('lon',))
        self.lons[:] = np.arange(-180,180,1)
        self.selectedData = self.nc.createVariable('data','f4',('time','lev','lat','lon'))
        self.dataRows = 0


        #获取数据
        ncfiles = glob(os.path.join(self.dataPath,'*'+self.parameters['var']+'*.nc'))
        if not ncfiles:
            print('Files Not Found!')
            return
        else:
            #数据文件按leadtime筛选，提取相关数据
            #文件名示例:20030701.atm.PREC.200307-200407_sfc_member.nc
            for filename in ncfiles:
                print(filename)

                basename = os.path.basename(filename)

                #读取，添加数据
                tmpfile = netCDF4.Dataset(filename,'r')
                self.selectedData[self.dataRows,:,:,:] = np.expand_dims(tmpfile.variables[self.parameters['var']][self.parameters['lt'],:,:,:],axis=0)
                tmpfile.close()

                #更新时间
                startDate = dt.datetime.strptime(basename.split('.')[0],"%Y%m%d")
                tmpMonth = startDate.month+self.parameters['lt']
                tmpYear = startDate.year
                if tmpMonth>12:
                    tmpMonth %= 12
                    tmpYear += tmpMonth/12
                self.monthList[tmpMonth].append(self.dataRows)
                self.times[self.dataRows] = '-'.join([str(tmpYear),str(tmpMonth)])

                self.dataRows += 1

            #据平
            self.cutMean()

            #保存数据
            self.nc.close()


    def cutMean(self):
        """数据基于月份的据平"""
        self.monthMean = list()
        for month in self.monthList:
            self.monthMean.append(np.mean(self.selectedData[self.monthList[month],:,:,:],axis=0))
        print(self.monthMean)
        # f = open('test.txt','w')
        # for idx,item in enumerate(self.monthMean):
        #     f.write(' '.join([str(idx+1),str(item)]))
        #     f.write('\n')
        # f.close()

        for idx in range(self.dataRows):
            month = int(self.times[idx].split('-')[1])
            self.selectedData[idx,:,:,:] -= self.monthMean[month-1]


if __name__ == '__main__':
    temp = CSM()
