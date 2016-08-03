#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from command import Argument, Settings
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
mpl.use('Agg')


class ROC(Argument, Settings):
    """ROC绘图"""
    parameters = dict()
    storePath = '.'
    n_classes = 3
    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('-ex',), {'help': u'样本文件路径', 'type': str, 'nargs': 1}],
        [('-o',), {'help': u'结果文件保存路径', 'type': str, 'nargs': 1}],
        [('-name',), {'help': u'结果文件名', 'type': str, 'nargs': 1}]
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        if self.args.ex:
            if os.path.isfile(self.args.ex[0]):
                self.parameters['ex'] = self.args.ex[0]
        if self.args.o:
            self.storePath = self.args.o[0]
        if self.args.name:
            self.parameters['name'] = self.args.name[0]
        self.process()

    def process(self):
        # 读取数据
        data = pd.DataFrame.from_csv(self.parameters['ex'])
        self.y_score = data[['pre_below', 'pre_normal', 'pre_above']]
        self.y_true = data[['obs_below', 'obs_normal', 'obs_above']]
        # 绘图
        fpr = dict()  # False Positive Rate
        tpr = dict()  # True Positive Rate
        # turn off the interactive mode
        plt.clf()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(
                self.y_true.ix[:, i], self.y_score.ix[:, i])
            plt.plot(fpr[i], tpr[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.parameters['name'])


if __name__ == '__main__':
    temp = ROC()
