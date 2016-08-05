#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from sys import exit
from command import Argument, Settings
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
import tempfile as tf
# mpl.use('Agg')


class ROC(Argument, Settings):
    """ROC绘图"""
    parameters = dict()
    storePath = '.'
    n_classes = 3
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
        # 读取数据
        data = pd.DataFrame.from_csv(self.parameters['ex'])
        self.y_score = data[['pre_below', 'pre_normal', 'pre_above']]
        self.y_true = data[['obs_below', 'obs_normal', 'obs_above']]
        # 绘图
        fpr = dict()  # False Positive Rate
        tpr = dict()  # True Positive Rate
        roc_auc = dict() #ROC AREA UNDER CURVE
        # turn off the interactive mode
        plt.clf()
        fpr[self.parameters['index']], tpr[self.parameters['index']], _ = metrics.roc_curve(self.y_true.ix[:, self.parameters['index']], self.y_score.ix[:, self.parameters['index']])
        roc_auc[self.parameters['index']] = metrics.roc_auc_score(self.y_true.ix[:, self.parameters['index']], self.y_score.ix[:, self.parameters['index']])
        if self.args.verbose:
            print("====False Positive Ratio(fpr) And True Positive Ratio(tpr) Pair====")
            for idx,val in enumerate(fpr[self.parameters['index']]):
                print(idx,val,fpr[self.parameters['index']][idx])
        plt.plot(fpr[self.parameters['index']], tpr[self.parameters['index']],label='ROC curve (area = %0.2f)' % roc_auc[self.parameters['index']])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.args.title[0] if self.args.title else 'Receiver Operating Characteristic(ROC)')
        plt.legend(loc="lower right")
        print('saving image to {}'.format(self.parameters['name']))
        plt.savefig(self.parameters['name'])
        print('Completely Finshed.')


if __name__ == '__main__':
    temp = ROC()
