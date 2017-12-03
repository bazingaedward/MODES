#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author: 邱凯翔<kxqiu@chinkun.cn>
latest: 2016.08.17
description: typhoon.py 台风数据处理及预测
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
import stormtracks as st


class Typhoon(Argument, Settings):
    """台风预测"""
    parameters = dict()
    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        # [('npy',), {'help': u'指定一个csv2npy后生成的.npy文件', 'type': str, 'nargs': 1}],
        [('-d','--download',), {'help': u'下载台风资料', 'choices':['C20','IBTRACKS']}],
        [('-p','--path',), {'help': u'设置下载路径,默认为当前路径', 'type':str, 'nargs':1}],

        [('type',), {'help': u'', 'choices':['download','analysis','utils'], 'nargs': 1}],
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
        pass



if __name__ == '__main__':
    temp = Typhoon()
