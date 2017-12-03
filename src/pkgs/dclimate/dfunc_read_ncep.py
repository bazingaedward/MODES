# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import netCDF4 as nc4
import numpy as np
import pandas as pd
import dfunc as dfunc
from  datetime import datetime
import os

def Get_NCEP_Day_List(path1=r'D:\NCEP_DAY', FHead='hgt', ilev=5, var_name='hgt', sdate1='1970-02-01', i_day_count=28,i_Year1=1982):
    '''
    E:\LAFCLI
    W:\DERF\LAFCLI
     从CFS历史和实时数据中获取文件名列表
    '''
    print('aa')
    dfunc.mytic()
    flist1 = dfunc.get_dir_list(path1, FHead)
    for list1 in flist1:
        print(list1)
    dfunc.mytoc()
    #sys.exit(0)
    NOW = datetime.now()
    i_current_year = int(datetime.strftime(NOW, '%Y'))

    I_Year = np.array(range(i_Year1, i_current_year))
    len1 = len(range(i_Year1, i_current_year))

    Field_All = np.zeros((len1, 73, 144))

    j = 0
    for i in range(i_Year1, i_current_year):
        #print(i)
        filename1 = os.path.join(path1, '%s.%d.nc' % (FHead, i))
        print(filename1, os.path.isfile(filename1))
        Field,dinfo = Read_Ncep_Day(filename1, ilev, var_name, sdate1, i_day_count)
        Field_All[j, :, :] = Field
        j = j + 1

    return Field_All,I_Year,dinfo


def Read_Ncep_Day(NcFileName=r'D:\NCEP_DAY\hgt.1970.nc', ilev=5, var_name='hgt', sdate1='1970-03-01', i_count=31):
    date1 = datetime.strptime(sdate1, '%Y-%m-%d')
    #####################################################################

    if(not  os.path.isfile(NcFileName)):
        print('%s not Exist'%NcFileName)
        return

    print('NcFileName=',NcFileName)
    rootgrp = nc4.Dataset(NcFileName, 'r')
    ##print(NcFileName+"  Format is: "+rootgrp.file_format)
    ##print(rootgrp.variables)

    #dinfo = {}

    #print('#'+'-'*79)
    #print(u'read %s 数据'%(NcFileName))
    #print('lev=',ilev,type(ilev))
    dinfo={}
    lat = rootgrp.variables['lat'][:]
    lon = rootgrp.variables['lon'][:]
    dinfo['lat']=lat
    dinfo['lon']=lon

    if 'level' in rootgrp.variables:
        level = rootgrp.variables['level'][:]
        dinfo["level"] = int(level[ilev])
    else:
        print("can't find level variable")

    times = rootgrp.variables['time']
    hgt = rootgrp.variables[var_name]

    dates = nc4.num2date(times[:], units=times.units)

    print(hgt.shape)
    L1 = 0
    for line1 in dates:
        #print(line1)
        if (datetime.strftime(line1, '%m-%d') == datetime.strftime(date1, '%m-%d')):
            break
        L1 = L1 + 1
    L2 = L1 + i_count
    print('L1=',L1,'L2=',L2)

    #print(date1)
    #print(dates)

    if 'level' in rootgrp.variables:
    #如果包含层，则使用层信息
        Field = np.mean(hgt[L1:L2, ilev, :, :], axis=0)
    else:
    #如果不包含层，则不是使用层信息
        Field = np.mean(hgt[L1:L2, :, :], axis=0)
    print(Field.shape)
    rootgrp.close()
    return Field,dinfo
    #sys.exit(0)
    #print(var_name+' shape is:',end='')  print(hgt.shape)   print(type(hgt)) sys.exit(0)

if __name__ == '__main__':
    path1 = r'd:\NCEPDAY'
    sdate1 = '1980-01-01'
    i_count = 30
    I_Year1 = 1980
    H500, I_Year,dinfo = Get_NCEP_Day_List(path1, 'hgt', 5, 'hgt', sdate1, i_count,I_Year1) #i_lev=6 第6层
    print(H500.shape)

    H500_avg = H500[0:30,:,:]

    H500_avg = np.mean(H500,axis=0)
    print(H500_avg.shape)

    lat = dinfo['lat']
    lon = dinfo['lon']

    import dclimate.dplotlib as dplot
    dplot.drawhigh(H500_avg)



