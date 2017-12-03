#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from sys import exit
from command import Argument, Settings
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import tempfile as tf
import numpy as np

class CD(Argument, Settings):
    """Climate Division"""
    parameters = dict()
    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('ini',), {'help': u'初始化参数文件', 'type': str, 'nargs': 1}],
        [('--clustering',), {'help': u'选取聚类类型', 'choices': ['REOF','AP'], 'nargs': 1}],
        [('-v','--verbose',), {'help': u'输出详细信息', 'action':"store_true"}]
    ]

    def __init__(self):
        """init"""
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        """parse arguments"""
        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,self.parameters[key])
        self.process()

    def process(self):
        pass

    def D1_05_get_AP_Cluster(FileName1=r'OBS_PAP_1981_2013_6_3_182.txt',NUM_CLUSTER=14):
        #RegionR = np.loadtxt(r'1971_2012_52_6_3_R.txt')
        RegionR = np.loadtxt(FileName1)
        Region = RegionR[1:,1:]
        Region_Obs = RegionR[1:,3:]
        I_Year = RegionR[0,3:]
        LonLat = RegionR[1:,1:3]
        LonLat2 = RegionR[1:,0:3]


        import  pkgs.dclimate.dcluster as dcluster
        #import distance_pearson_corr
        X=RegionR[1:,3:]
        #X = np.random.random([6,5])
        #pmat = pdist(X, "euclidean")
        #自定义距离函数
        #print(LonLat)
        #sys.exit(0)

        #### dis2_euclidean = distance.pdist(LonLat)

        #print(dis2_euclidean,np.max(dis2_euclidean))
        #dis2_euclidean = np.where(dis2_euclidean<4,0,dis2_euclidean)

        #### dis2_euclidean = dis2_euclidean/40

        #print(dis2_euclidean,np.max(dis2_euclidean))
        #sys.exit(0)

        dis1_corr = distance.pdist(X,lambda u,v:1-dcluster.distance_pearson_corr(u,v) )
        #print(np.min(dis2_euclidean))
        #print(np.max(dis2_euclidean))
        #sys.exit(0)
        #pmat=dis1_corr#+dis2_euclidean
        pmat=dis1_corr
        S = distance.squareform(pmat)

        S=-S
        P=np.median(S)

        #################################
        if(NUM_CLUSTER<=0):
            idx,k=dcluster.apclusterd(S)
        else:
            idx,k=dcluster.apcluster_k(S,NUM_CLUSTER)

        #sys.exit(0)
        #################################
        print('idx=',idx)
        print('numbers ',k)

        id1 = np.unique(idx)
        idx2=idx
        print(idx2)
        print(id1)

        for ii in range(len(id1)):
            print('%d %d'%(ii,id1[ii]))
            aa = np.where(idx2 == id1[ii],True,False)
            #print(idx2)
            #print(aa)
            idx2[aa]=ii
            #id1[ii]
        #sys.exit(0)
        idx2=idx2+1
        print(S.shape)
        print(idx2)
        out1 = np.vstack((LonLat2.T,idx2))
        out1 = out1.T
        FileName='ap_cluster.txt'
        np.savetxt(FileName,out1,fmt='%5d %6.2f %6.2f %d')
        return FileName



if __name__ == '__main__':
    temp = CD()
