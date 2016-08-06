#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author: qkx
latest: 2016.08.06
"""
from __future__ import print_function
from sys import exit
from command import Argument, Settings
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import tempfile as tf

class GridPlot(Argument, Settings):
    """ROC绘图"""
    parameters = dict()
    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('ex',), {'help': u'指定一个序列化后的样本文件', 'type': str, 'nargs': 1}],
        [('index',), {'help': u'输出三分类中的哪一层:0:below; 1:normal; 2:above', 'type': int, 'nargs': 1}],
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
        if self.args.ex and os.path.isfile(self.args.ex[0]):
            self.parameters['ex'] = self.args.ex[0]
        else:
            print('{} not found!'.format(self.args.ex[0]))
            exit(-1)
        self.parameters['name'] = self.args.name[0] if self.args.name else tf.mktemp(suffix='.png',dir='.')
        self.parameters['index'] = self.args.index[0] if self.args.index else 0
        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,self.parameters[key])
        self.process()

    def process(self):
        """ process """
        pass


if __name__ == '__main__':
    temp = GridPlot()
