# -*- coding: cp936 -*-
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

#######################################################################
#绘制588高度场曲线
#######################################################################
def draw_588_Line(H500):
    lev1 = np.array([5840,5880])
    FieldP_avg = np.mean(H500[0:-1,:,:],axis=0)
    FieldP_end =H500[-1,:,:]
    #黑线为平均值
    #红线为预测值
    ptitle=' 588 line avg and real'
    lons = np.arange(0, 360, 2.5, dtype=float)
    lats = np.arange(90, -90 - 1, -2.5, dtype=float)
    dplot.drawhigh5880Line(FieldP_end,FieldP_avg,lons,lats,ptype=1, \
                           ptitle=ptitle, \
                           imgfile='018PreNCP_588line.png',showimg=1,lev=lev1)


def read_necp_mon(Nc_Path,I_Year,Mon,Months_Count,datefile='hgt_sel_ncep.npy'):
    #I_Year  = np.linspace(1982,2012,2012-1982+1)
    #Mon = 6
    #Months_Count = 3
    hgt_file=os.path.join(Nc_Path,'hgt.mon.mean.nc')
    H500,dinfo=df.Read_Ncep_Hgt(hgt_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='hgt',ilev=5)
    #draw_588_Line(H500)
    H200,dinfo=df.Read_Ncep_Hgt(hgt_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='hgt',ilev=9)
    H700,dinfo=df.Read_Ncep_Hgt(hgt_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='hgt',ilev=3)

    uwnd_file = os.path.join(Nc_Path,'uwnd.mon.mean.nc')
    U700,dinfo=df.Read_Ncep_Hgt(uwnd_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='uwnd',ilev=2)
    U200,dinfo=df.Read_Ncep_Hgt(uwnd_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='uwnd',ilev=9)

    vwnd_file = os.path.join(Nc_Path,'vwnd.mon.mean.nc')
    V700,dinfo=df.Read_Ncep_Hgt(vwnd_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='vwnd',ilev=2)
    V200,dinfo=df.Read_Ncep_Hgt(vwnd_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='vwnd',ilev=9)

    slp_file = os.path.join(Nc_Path,'SLP.mon.mean.nc')
    SLP,dinfo=df.Read_Ncep_Hgt(slp_file,I_Year,Mon,Months_Count,FieldOffset=0,var_name='slp',ilev=9)

    dict1 = {}
    #dict1['xmin'],dict1['xmax']=x1,x2
    #dict1['H700'] = H700
    dict1['H500'] = H500
    dict1['H200'] = H200
    dict1['U200'] = U200
    dict1['U700'] = U700
    dict1['V200'] = V200
    dict1['V700'] = V700
    dict1['SLP'] = SLP

    dict1['I_Year'] = I_Year
    dict1['Mon']=3
    dict1['I_COUNT']=Months_Count
    #Get_NCEP_Day_List(path1=r'D:\NCEP_DAY',FHead ='slp',ilev=5,var_name='slp',sdate1='1970-03-01',i_count = 31)
    #print(np.linspace(1982,2012,2012-1982+1))
    df.save_obj(dict1, datefile)


def read_model_pred_nc(ncFileName,I_Year,Mon,Months_Count,datefile='hgt_sel_pred.npy'):
    #I_Year  = np.linspace(1982,2013,2013-1982+1)
    #Mon = 4
    #Months_Count = 1

    #ncFileName = r'E:\BDATA3\py301_listcfs\cfs031200.nc'
    H500,dinfo=df.Read_Ncep_Hgt(ncFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='z500',ilev=5)

    #print(H500)
    #sys.exit(0)
    H200,dinfo=df.Read_Ncep_Hgt(ncFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='z200',ilev=5)
    #H700=df.Read_Ncep_Hgt(r'cfs032200.nc',I_Year,Mon,Months_Count,FieldOffset=0,var_name='z500',ilev=5)

    U700,dinfo=df.Read_Ncep_Hgt(ncFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='u850',ilev=5)
    U200,dinfo=df.Read_Ncep_Hgt(ncFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='u200',ilev=9)

    V700,dinfo=df.Read_Ncep_Hgt(ncFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='v850',ilev=5)
    V200,dinfo=df.Read_Ncep_Hgt(ncFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='v200',ilev=9)

    SLP,dinfo=df.Read_Ncep_Hgt(ncFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='precsfc',ilev=9)

    dict1={}
    #dict1['xmin'],dict1['xmax']=x1,x2
    #dict1['H700']=H700
    dict1['H500']=H500
    dict1['H200']=H200
    dict1['U200']=U200
    dict1['U700']=U700
    dict1['V200']=V200
    dict1['V700']=V700
    dict1['SLP']=SLP


    dict1['I_Year']=I_Year
    dict1['I_Mon']=Mon
    #dict1['I_Hou']=int(I_Hou)


    print(I_Year)
    #import dclimate.dfunc as df
    df.save_obj(dict1,datefile)
    #SLP=df.Read_Ncep_Hgt(r'd:\ncepmon\SLP.mon.mean.nc',I_Year,Mon,Months_Count,FieldOffset=0,var_name='precsfc',ilev=9)


if __name__ == '__main__':
    ############################################
    #
    I_Year  = np.linspace(1982,2012,2012-1982+1)

    Mon,Months_Count=8,1 #4月 1个月
    read_necp_mon(I_Year,Mon,Months_Count)
    #sys.exit(0)
    ############################################
    #ncFileName = r'E:\BDATA3\py301_listcfs\cfs072000.nc'
    ncFileName = r'E:\BDATA3\py301_listTCC\tcc0715.nc'


    #ncFileName = r'E:\BDATA3\py301_listcfs\ncc05.nc'
    I_Year  = np.linspace(1982,2013,2013-1982+1)

    if(0):
        a = I_Year==2011
        a = np.logical_not(a)
        I_Year =I_Year[a]

    read_model_pred_nc(ncFileName,I_Year,Mon,Months_Count)


    #cfs032200

