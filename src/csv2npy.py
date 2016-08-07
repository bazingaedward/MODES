#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author: 邱凯翔<kxqiu@chinkun.cn>
latest: 2016.08.07
description: csv2npz.py是格式转换工具,将一个或多个序列化后的文件合并处理后保存成统一的.npz格式.
"""
from __future__ import print_function
from sys import exit
from command import Argument, Settings
import pandas as pd
from sklearn import metrics
import tempfile as tf
import glob
import os
import numpy as np

class csv2npy(Argument, Settings):
    """将一个或多个序列化后的文件合并处理后保存成统一的.npz格式"""
    parameters = dict()
    lats = 181
    lons = 360
    nclass = {0:'below',1:'normal',2:'above'}
    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('csv_dir',), {'help': u'指定一个序列化后的样本文件', 'type': str, 'nargs': 1}],
        [('index',), {'help': u'输出三分类中的哪一层:0:below; 1:normal; 2:above', 'type': int, 'nargs': 1}],
        [('-n','--name',), {'help': u'输出结果文件名', 'type': str, 'nargs': 1}],
        [('-v','--verbose',), {'help': u'输出详细信息', 'action':"store_true"}]
    ]

    def __init__(self):
        """init"""
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        """parse arguments"""
        if self.args.csv_dir and os.path.isdir(self.args.csv_dir[0]):
            self.parameters['csv_dir'] = self.args.csv_dir[0]
        else:
            print('{} not found!'.format(self.args.csv_dir[0]))
            exit(-1)
        self.parameters['name'] = self.args.name[0] if self.args.name else tf.mktemp(suffix='.npz',dir='.')
        self.parameters['index'] = self.args.index[0] if self.args.index else 0

        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,self.parameters[key])
        self.process()

    def process(self):
        """ process """
        ##directory check
        files = glob.glob(os.path.join(self.parameters['csv_dir'],'*.csv'))

        if not files:
            print('No .csv file found in {}.'.format(self.parameters['csv_dir']))
            exit(-1)

        self.auc = np.zeros([self.lats,self.lons])
        self.bs = np.zeros([self.lats,self.lons])

        ##loop for reshape data
        for lat in np.arange(self.lats):
            for lon in np.arange(self.lons):
                if self.args.verbose:
                    print('Now Calculating Grid({},{})......'.format(lat,lon))
                y_true = list()
                y_score = list()
                rowIdx = lat*self.lons+lon
                for path in files:
                    row = pd.DataFrame.from_csv(path).iloc[rowIdx]
                    y_true.append(row['obs_'+self.nclass[self.parameters['index']]])
                    y_score.append(row['pre_'+self.nclass[self.parameters['index']]])

                ##校验y_true结果,如果全是0则跳过后面的计算
                if all(i == 0 for i in y_true):
                    print('Warning:Grid({},{} y_true has only one class(0 or 1))'.format(lat,lon))
                    continue

                ##计算auc,bs
                self.auc[lat,lon] = metrics.roc_auc_score(y_true,y_score)
                self.bs[lat,lon] = metrics.brier_score_loss(y_true,y_score)
                print(self.auc[lat,lon],self.bs[lat,lon])
                del(y_true)
                del(y_score)

        ##save result
        np.save(self.parameters['name']+'_auc',self.auc)
        np.save(self.parameters['name']+'_bs',self.bs)


if __name__ == '__main__':
    temp = csv2npy()