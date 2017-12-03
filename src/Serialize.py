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
import tempfile as tf


class Serialization(Argument, Settings):
    """针对预处理后的样本序列化"""
    parameters = dict()
    grid = {'lat': 181, 'lon': 360}  # 网格尺寸，观测与预报的必须统一

    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('pre',), {'help': u'概率预测文件路径', 'type': str, 'nargs': 1}],
        [('obs',), {'help': u'观测文件路径', 'type': str, 'nargs': 1}],
        [('-t','--type',), {'help': u'文件保存方式,默认single方式', 'choices':['single','multiple'], 'nargs': 1}],
        [('-n','--name',), {'help': u'结果文件名', 'type': str, 'nargs': 1}],
        [('-v','--verbose',), {'help': u'输出详细信息', 'action':"store_true"}]
    ]

    # 序列化导出数据表单,每条记录针对格点数据
    columns = ['pre_below', 'pre_normal', 'pre_above', 'obs_below',
               'obs_normal', 'obs_above', 'latitude', 'longitude']
    result = pd.DataFrame(columns=columns)

    def __init__(self):
        """init"""
        Argument.__init__(self)
        Settings.__init__(self)
        self.startTimer = time.time()
        self.parse_args()

    def parse_args(self):
        """parse arguments"""
        if self.args.pre and os.path.isfile(self.args.pre[0]):
            self.parameters['pre'] = self.args.pre[0]
        else:
            print('{} not found!'.format(self.args.pre[0]))
            exit(-1)

        if self.args.obs and os.path.isfile(self.args.obs[0]):
            self.parameters['obs'] = self.args.obs[0]
        else:
            print('{} not found!'.format(self.args.obs[0]))
            exit(-1)

        self.parameters['name'] = self.args.name[0] if self.args.name else tf.mktemp(suffix='.csv',dir='.')
        self.parameters['type'] = self.args.type[0] if self.args.type else 'single'
        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,self.parameters[key])
            print("=================================")
        self.process()

    def process(self):
        """process"""
        if self.args.verbose:
            print('Reading Data Files......')
        # 概率预测数据
        pre = netCDF4.Dataset(self.parameters['pre'], 'r')
        self.Time_pre = pre.variables['times'][:]
        self.Data_pre = pre.variables['data'][:]
        # 观测数据
        obs = netCDF4.Dataset(self.parameters['obs'], 'r')
        self.Time_obs = obs.variables['time'][:]
        self.Data_obs = obs.variables['obcat'][:]
        # 数据匹配
        self.startTime = dt.datetime.strptime(
            str(self.Time_obs[0]), '%Y%m')  # 顺序排列的观测数据起始月份
        # 确定三分类30%和70%的阈值分割点
        if self.args.verbose:
            print(u'计算三分类30%和70%的阈值分割点')
        self.cutoff = [np.percentile(
            self.Data_pre, 30), np.percentile(self.Data_pre, 70)]
        if self.parameters['type'] == 'single':
            self.singleSave()
        else:
            self.multipleSave()
        print("Total costs %s minutes." %
              (int(time.time() - self.startTimer) / 60))
        #close dataset
        pre.close()
        obs.close()

    def singleSave(self):
        """将所有数据保存到单一文件中"""
        for idx, val in enumerate(self.Time_pre):
            if self.args.verbose:
                print('Now Processing {} date.......'.format(val))
            # tmpDate:观测时间中的每个值
            tmpDate = dt.datetime.strptime(val, '%Y-%m')
            # diffMonth:tmpDate-startTime的月份差，作为读取观测数据的序数
            diffMonth = (tmpDate.year - self.startTime.year) * \
                12 + (tmpDate.month - self.startTime.month)
            for lat in range(self.grid['lat']):
                for lon in range(self.grid['lon']):
                    list_pre = self.calProb(self.Data_pre[idx, :, lat, lon])
                    list_obs = self.calObs(self.Data_obs[diffMonth, lat, lon])
                    ## 检查list_obs结果
                    if not list_obs:
                        continue
                    ## 结果保存
                    df = pd.DataFrame(
                        [list(list_pre) + list_obs + [lat, lon]],
                        columns=self.columns)
                    # print(df)
                    self.result = pd.concat([self.result, df])
            # 显示进度
            print("Complete %s %%" % (idx * 100.0 / len(self.Time_pre)))

        # 保存结果，默认以csv格式保存
        print('Saving data to {}'.format(self.parameters['name']))
        self.result.to_csv(self.parameters['name'])

    def multipleSave(self):
        """将数据按月保存"""
        for idx, val in enumerate(self.Time_pre):
            if self.args.verbose:
                print('Now Processing {} date.......'.format(val))
            #clear results
            self.result = None
            # tmpDate:观测时间中的每个值
            tmpDate = dt.datetime.strptime(val, '%Y-%m')
            # diffMonth:tmpDate-startTime的月份差，作为读取观测数据的序数
            diffMonth = (tmpDate.year - self.startTime.year) * \
                12 + (tmpDate.month - self.startTime.month)
            for lat in range(self.grid['lat']):
                for lon in range(self.grid['lon']):
                    list_pre = self.calProb(self.Data_pre[idx, :, lat, lon])
                    list_obs = self.calObs(self.Data_obs[diffMonth, lat, lon])
                    ## 检查list_obs结果
                    if not list_obs:
                        continue
                    ## 结果保存
                    df = pd.DataFrame(
                        [list(list_pre) + list_obs + [lat, lon]],
                        columns=self.columns)
                    # print(df)
                    self.result = pd.concat([self.result, df])
            # 显示进度
            print("Complete %s %%" % (idx * 100.0 / len(self.Time_pre)))

            # 保存结果，默认以csv格式保存
            url = self.parameters['name']+'_'+val+'.csv'
            print('Saving data to {}'.format(url))
            self.result.to_csv(url)


    def calProb(self, data):
        """ 计算CSM的24个模式概率 """
        result = []
        for item in data:
            if item < self.cutoff[0]:
                result.append([1, 0, 0])
            elif item < self.cutoff[1]:
                result.append([0, 1, 0])
            else:
                result.append([0, 0, 1])
        return np.sum(result, axis=0) / float(len(data))


    def calObs(self, data):
        """ 观测结果 """
        if data == 0:
            return [1, 0, 0]
        elif data == 1:
            return [0, 1, 0]
        elif data == 2:
            return [0, 0, 1]
        else:
            return False


if __name__ == '__main__':
    temp = Serialization()