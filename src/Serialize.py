#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from command import Argument, Settings
import os
import netCDF4
import datetime as dt
import numpy as np
import pandas as pd
import time

class Serialization(Argument, Settings):
    """针对预处理后的样本序列化"""
    parameters = dict()
    storePath = '.'
    grid = {'lat':181,'lon':360}#网格尺寸，观测与预报的必须统一

    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('-pre',), {'help': u'概率预测文件路径', 'type': str, 'nargs':1}],
        [('-obs',), {'help': u'观测文件路径', 'type': str, 'nargs':1}],
        [('-o',), {'help': u'结果文件保存路径', 'type': str, 'nargs':1}],
        [('-name',), {'help': u'结果文件名', 'type': str, 'nargs':1}]
    ]

    #序列化导出数据表单,每条记录针对格点数据
    columns = ['pre_below','pre_normal','pre_above','obs_below','obs_normal',\
               'obs_above','latitude','longitude']
    result = pd.DataFrame(columns=columns)

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)
        self.startTimer = time.time()
        self.parse_args()

    def parse_args(self):
        if self.args.pre:
            if os.path.isfile(self.args.pre[0]):
                self.parameters['pre'] = self.args.pre[0]

        if self.args.obs:
            if os.path.isfile(self.args.obs[0]):
                self.parameters['obs'] = self.args.obs[0]

        if self.args.o:
            self.storePath = self.args.o[0]

        if self.args.name:
            self.parameters['name'] = self.args.name[0]

        self.process()

    def process(self):
        #概率预测数据
        pre = netCDF4.Dataset(self.parameters['pre'],'r')
        Time_pre = pre.variables['times'][:]
        Data_pre = pre.variables['data'][:]
        #观测数据
        obs = netCDF4.Dataset(self.parameters['obs'],'r')
        Time_obs = obs.variables['time'][:]
        Data_obs = obs.variables['obpre'][:]
        #数据匹配
        startTime = dt.datetime.strptime(str(Time_obs[0]),'%Y%m')#顺序排列的观测数据起始月份
        #确定三分类30%和70%的阈值分割点
        self.cutoff = [np.percentile(Data_pre,30),np.percentile(Data_pre,70)]
        for idx,val in enumerate(Time_pre[:1]):
            #tmpDate:观测时间中的每个值
            tmpDate = dt.datetime.strptime(val,'%Y-%m')
            #diffMonth:tmpDate-startTime的月份差，作为读取观测数据的序数
            diffMonth = (tmpDate.year-startTime.year)*12+(tmpDate.month-startTime.month)
            for lat in range(self.grid['lat']):
                for lon in range(self.grid['lon']):
                    list_pre = self.calProb(Data_pre[idx,:,lat,lon])
                    list_obs = self.calObs(Data_obs[diffMonth,lat,lon])
                    #结果保存
                    df = pd.DataFrame([list(list_pre)+list_obs+[lat,lon]],columns=self.columns)
                    # print(df)
                    self.result = pd.concat([self.result,df])
            #显示进度
            print("Complete %s %%" %(idx*100.0/len(Time_pre)))

        #保存结果，默认以csv格式保存
        print('Saving to file......')
        url = os.path.join(self.storePath,self.parameters['name'])
        self.result.to_csv(url)
        print("Total costs %s minutes." %(int(time.time()-self.startTimer)/10))

    #计算CSM的24个模式概率
    def calProb(self,data):
        result = []
        for item in data:
            if item < self.cutoff[0]:
                result.append([1,0,0])
            elif item < self.cutoff[1]:
                result.append([0,1,0])
            else:
                result.append([0,0,1])
        return np.sum(result,axis=0)/float(len(data))

    #观测结果
    def calObs(self,data):
        if data == -1:
            return [1,0,0]
        elif data == 0:
            return [1,0,0]
        else:
            return [0,0,1]

if __name__ == '__main__':
    temp = Serialization()

