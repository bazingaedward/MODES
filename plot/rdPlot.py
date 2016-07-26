#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from glob import glob
from command import Argument, Settings
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from sklearn import metrics

class Verification(Argument, Settings):
    """ROC/BSS检验类"""
    parameters = dict()#命令所获取的所有参数
    relatedFiles = []#给定预报时间后目录下在该范围内的文件名列表
    relatedData = []#给定预报时间后目录下在该范围内的数据
    dataFlag = 0#标记relatedData是否第一次处理
    n_classes = 3

    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('-data',), {'help':'xxxx'}],
        [('-mod',), {'help': u'模式', 'choices': ['CSM', 'POAMA', 'UKMO']}],
        [('-lon',), {'help': u'经度', 'type': float, 'nargs':2}],
        [('-lat',), {'help': u'纬度', 'type': float, 'nargs':2}],
        [('-e',), {'help': u'变量(Element)', 'type': str, 'nargs':1}],
        [('-event',), {'help': u'事件', 'choices': ['below', 'normal', 'above']}],
        [('-it',), {'help': u'起报时间(initial time)', 'type': str, 'nargs':1}],
        [('-lt',), {'help': u'预报时间(lead time)', 'type': int, 'nargs':1}]
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)

        self.parse_args()

    def parse_args(self):
        #模式参数
        if self.args.mod:
            mod = self.args.mod.upper() if self.args.mod is not None else ''
            self.parameters['mod'] = mod

        #起报时刻
        if self.args.it:
            it = dt.datetime.strptime(self.args.it[0],'%Y-%m')
            self.parameters['it'] = it

        #预报时刻
        if self.args.lt:
            lt = self.args.lt[0]
            self.parameters['lt'] = lt

        #变量
        if self.args.e:
            self.parameters['e'] = self.args.e[0]

        if mod == 'CSM':
            self.csm()
        elif mod == 'POAMA':
            self.poama()

    def csm(self):
        #获取数据
        csmpath = self.getcfg('models', 'csm')
        ncfiles = glob(os.path.join(csmpath,'*'+self.parameters['e']+'*.nc'))
        if ncfiles:
            #初始时刻+预报时刻
            months = self.parameters['it'].month+self.parameters['lt']
            if months>12:
                year = self.parameters['it'].year+months/12
                month = months%12
                date = self.parameters['it'].replace(year=year,month=month)
            else:
                date = self.parameters['it'].replace(month=months)

            #预报时刻标识，用以判断当前数据的在relatedData中的位置
            dateString = self.parameters['it'].strftime("%Y%m%d")

            #数据文件按时间筛选，提取相关数据
            #文件名示例:20030701.atm.PREC.200307-200407_sfc_member.nc
            for filename in ncfiles:
                basename = os.path.basename(filename)
                startDate = dt.datetime.strptime(basename.split('.')[0],"%Y%m%d")
                tmpfile = netCDF4.Dataset(filename,'r')
                tmpMonth = startDate.month+self.parameters['lt']
                if tmpMonth>12:
                     tmpMonth %= 12
                if tmpMonth == date.month:
                    #加入文件列表
                    self.relatedFiles.append(filename)
                    if self.dataFlag:
                        self.relatedData = np.concatenate((self.relatedData,np.expand_dims(tmpfile.variables[self.parameters['e']][self.parameters['lt'],:,:,:],axis=0)))
                    else:
                        self.dataFlag = 1
                        self.relatedData = np.expand_dims(tmpfile.variables[self.parameters['e']][self.parameters['lt'],:,:,:],axis=0)
                    #计算lead time
            #         timeIdx = (date.year-startDate.year)*12+(date.month-startDate.month)
            #         if self.dataFlag:
            #             self.relatedData = np.concatenate((self.relatedData,np.expand_dims(tmpfile.variables[self.parameters['e']][timeIdx,:,:,:],axis=0)))
            #         else:
            #             self.dataFlag = 1
            #             self.relatedData = np.expand_dims(tmpfile.variables[self.parameters['e']][timeIdx,:,:,:],axis=0)
            #
            #         #当前数据
            #         if dateString in filename:
            #             self.currentData = tmpfile.variables[self.parameters['e']][timeIdx,:,:,:]
            #锯平
            mean = np.mean(self.relatedData,axis=0)
            self.relatedData -= mean

            #确定30%和70%的分割点
            self.cutoff = [np.percentile(self.relatedData,30),np.percentile(self.relatedData,70)]

            #计算三分类概率
            self.y_score = []
            shape = self.relatedData.shape
            for lat in range(shape[2]):
                for lon in range(shape[3]):
                    self.y_score.append(self.calProb(np.reshape(self.relatedData[:,:,lat,lon],-1)))
            self.y_score = np.array(self.y_score)

            #读取观测数据
            obsFile = netCDF4.Dataset(os.path.join('/data/modes/mod','obpre.nc'),'r')
            timeIdx = (date.year-1990)*12 + (date.month-7)
            obsData = obsFile.variables['obpre'][timeIdx,:,:]
            self.y_true = []
            shape = obsData.shape
            for lat in range(shape[0]):
                for lon in range(shape[1]):
                    if obsData[lat,lon] == -1:
                        self.y_true.append([1,0,0])
                    elif obsData[lat,lon] == 0:
                        self.y_true.append([0,1,0])
                    elif obsData[lat,lon] == 1:
                        self.y_true.append([0,0,1])
                    else:
                        self.y_true.append([1,0,0])
            self.y_true = np.array(self.y_true)

            #ROC,BSS绘图
            self.rocPlot()

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

    def poama(self):
        print('poama')

    def rocPlot(self):
        fpr = dict()#False Positive Rate
        tpr = dict()#True Positive Rate
        roc_auc = dict()#Area Under the Curve (AUC)
        bss = dict()#Brier Score Skill
        refx = np.array([0.3,0.4,0.3])
        ref = np.tile(refx,(len(self.y_true),1))
        #turn off the interactive mode
        plt.ioff()
        text = ['below','normal','above']
        for i in range(self.n_classes):
            fpr[i],tpr[i],_ = metrics.roc_curve(self.y_true[:,i],self.y_score[:,i])
            roc_auc[i] = metrics.roc_auc_score(self.y_true[:,i],self.y_score[:,i])
            bss[i] = 1-metrics.brier_score_loss(self.y_true[:,i],self.y_score[:,i])/metrics.brier_score_loss(self.y_true[:,i],ref[:,i])
            plt.plot(fpr[i], tpr[i], label='class {0}(area = {1:0.2f},BSS = {2:0.2f})'.format(text[i], roc_auc[i],bss[i]))

        #绘图
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc2.png')


if __name__ == '__main__':
    temp = Verification()
