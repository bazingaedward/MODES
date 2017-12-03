#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author: 邱凯翔<kxqiu@chinkun.cn>
latest: 2016.08.15
description:  CMA.py基于气象中心全国站点日资料分析类
data_source:  中国地面气候资料日值数据集(V3.0)
data_created: 2012-08-04
data_ID:      SURF_CLI_CHN_MUL_DAY
data_version: v3.0
"""
from __future__ import print_function
from sys import exit
from command import Argument, Settings
import os
import pandas as pd
import tempfile as tf


class CMA(Argument, Settings):
    """CMA数据分析"""
    parameters = dict()
    PREFIX = 'SURF_CLI_CHN_MUL_DAY'
    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('data',), {'help': u'指定读入数据的文件名', 'type': str, 'nargs': 1}],
        [('-v','--verbose',), {'help': u'输出详细信息', 'action':"store_true"}]
    ]

    def __init__(self):
        """init"""
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        """parse arguments"""
        if self.args.data and os.path.isfile(self.args.data[0]):
            self.parameters['data'] = self.args.data[0]
        else:
            print('{} not found!'.format(self.args.data[0]))
            exit(-1)
        # self.parameters['name'] = self.args.name[0] if self.args.name else tf.mktemp(suffix='.png',dir='.')
        # self.parameters['title'] = self.args.title[0] if self.args.title else 'Grid Plot'

        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,self.parameters[key])
        self.process()

    def process(self):
        """ process """
        ## read data from file path
        self.data = pd.read_csv(self.parameters['data'],delim_whitespace=True,header=None)
        print(self.data)


if __name__ == '__main__':
    temp = CMA()