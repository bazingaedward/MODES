# -*- coding: cp936 -*-
'''
读取文件的集中式包，避免以前读取数据的库四处摆放的问题
创建：2014-05-29
修订：无
'''
from __future__ import print_function
from __future__ import division

import netCDF4 as nc4
import dplotlib as dplot
import numpy as np
from mpl_toolkits.basemap import Basemap, shiftgrid
import os, sys, re
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('AGG')
import dfunc as df
import py_derf02_readdat6 as derf6
from  datetime import datetime

#############################################################################
#读取模式数据
#############################################################################
def read_climate_model_from_path(ModelPath,ModelPatten,Mon_Init,Year1,Year2, \
                                 Mon_Pred,MonthCount,var_name1,Check_Year=False):
    #获取文件所有列表

    AllList1 = df.get_dir_all_file_list(ModelPath)
    #print(AllList1)
    re_str = ModelPatten
    print(re_str)
    #sys.exit(0)
    filelist1 = df.get_sel_list_from_all_list(AllList1,re_str)

    #print(filelist1)
    dict1 = {}
    List_Year_Model=[]
    Field_All=[]
    for i_Y in range(Year1,Year2):
        re_str = ModelPatten+'_'+'%4d'%i_Y+'%02d'%Mon_Init

        print(i_Y,re_str)
        filelist1 = df.get_sel_list_from_all_list(AllList1,re_str)
        print(filelist1)

        #print(len(filelist1))
        if(len(filelist1)>0):
            #sys.exit(0)
            #print(filelist1[0])

            List_Year_Model.append(i_Y)
            if(not Check_Year):
                filename1=filelist1[-1]
                print('Selected FileName =',filename1)
                Field1,lat,lon = read_climate_model_dat_One(filename1,i_Y,int(Mon_Pred),int(MonthCount),var_name=var_name1)

                print(Field1.shape,i_Y)
                dict1[i_Y]=Field1

    if(Check_Year):
        lat=None;lon=None
        return Field_All,List_Year_Model,lat,lon
    print(List_Year_Model)

    shape1 = np.shape(dict1[List_Year_Model[0]])
    Field_All=np.zeros( (len(List_Year_Model),shape1[0],shape1[1] ))

    print(shape1)
    for ii in range(len(List_Year_Model)):
        Tmp_Year = List_Year_Model[ii]
        Field_All[ii,:,:]=dict1[Tmp_Year]

    return Field_All,np.array(List_Year_Model),lat,lon


