#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author: 邱凯翔<kxqiu@chinkun.cn>
latest: 2016.08.06
description: GridPlot.py专门绘制网格格点图,并叠加地图信息
"""
from __future__ import print_function
from sys import exit
from command import Argument, Settings
import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import tempfile as tf
import numpy as np


class GridPlot(Argument, Settings):
    """ROC绘图"""
    parameters = dict()
    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('npy',), {'help': u'指定一个csv2npy后生成的.npy文件', 'type': str, 'nargs': 1}],
        [('-n','--name',), {'help': u'输出结果文件名', 'type': str, 'nargs': 1}],
        [('-t','--title',), {'help': 'plot image title', 'type': str, 'nargs': 1}],
        [('-v','--verbose',), {'help': u'输出详细信息', 'action':"store_true"}]
    ]

    def __init__(self):
        """init"""
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        """parse arguments"""
        if self.args.npy and os.path.isfile(self.args.npy[0]):
            self.parameters['npy'] = self.args.npy[0]
        else:
            print('{} not found!'.format(self.args.npy[0]))
            exit(-1)
        self.parameters['name'] = self.args.name[0] if self.args.name else tf.mktemp(suffix='.png',dir='.')
        self.parameters['title'] = self.args.title[0] if self.args.title else 'Grid Plot'

        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,self.parameters[key])
        self.process()

    def process(self):
        """ process """
        ##load data
        data = np.load(self.parameters['npy'])
        ##create basemap
        self.bm = Basemap(projection='cyl',resolution='l',lon_0=120)
        self.bm.drawcoastlines(linewidth=0.25)
        self.bm.drawcountries(linewidth=0.25)
        #self.bm.fillcontinents(color='grey')

        lons,lats = self.bm.makegrid(360,181)
        x,y = self.bm(lons,lats)
        self.bm.contourf(x,y,data)
        ##add colorbar
        self.bm.colorbar(location='bottom',size='5%',label="mm")
        ##add plot title
        plt.title(self.parameters['title'])

        ##save plot
        plt.savefig(self.parameters['name'])


if __name__ == '__main__':
    temp = GridPlot()
