# -*- coding: cp936 -*-
from __future__ import print_function
from __future__ import division

from dateutil.relativedelta import relativedelta
import datetime
import numpy as np
import pandas as pd
import dfunc as df
import netCDF4 as nc4
import sys,os
import d_std_lib as d_std_lib

#------------------------------------------------------------------------------
def cur_file_dir():
    #获取脚本路径
    path = sys.path[0]
    #判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)


def Get_Nc_VarName_List(ModelPath,ModelPatten):

    AllList1 = df.get_dir_all_file_list(ModelPath)
    #print(AllList1)
    re_str = ModelPatten
    print(re_str)
    #sys.exit(0)
    filelist1 = df.get_sel_list_from_all_list(AllList1,re_str)


    FieldFileName = filelist1[0]
    print('FieldFileName=',FieldFileName)
    rootgrp = nc4.Dataset(FieldFileName,'r')
    #print(rootgrp.file_format)
    #print(rootgrp.variables)

    var_dict = rootgrp.variables

    list1 = []
    for i in var_dict:
        print("var_dict['%s']=" % i, end='')#,dict2[i])
        #print(var_dict[i].shape)
        if(len(var_dict[i].shape)>1):
            print(i)
            list1.append(i)
            #sys.exit(0)
    rootgrp.close()

    return list1;




def read_climate_model_dat_One(FieldFileName,Specify_Year,Mon,Months_Count,var_name='z500'):
    print(FieldFileName)
    rootgrp = nc4.Dataset(FieldFileName,'r')
    print(rootgrp.file_format)
    #print(rootgrp.variables)

    #sys.exit(0)
    print('----Start read model data---- '*2)
    #sys.exit(0)
    lat = rootgrp.variables['lat'][:];
    lon = rootgrp.variables['lon'][:];
    #level = rootgrp.variables['level'][:];

    if 'level' in rootgrp.variables:
        level = rootgrp.variables['level'][:]
        dinfo["level"]=int(level[ilev])
    else:
        print("The NC file Can't find level variable")

    times = rootgrp.variables['time'];
    hgt = rootgrp.variables[var_name];

    #生成netcdf时间维数据
    nc_date_str = nc4.num2date(times[:],units=times.units)

    s1 ='%04d-%02d-01'%(Specify_Year,Mon)
    date1 = datetime.datetime.strptime(s1,'%Y-%m-%d')
    date2 = date1 + relativedelta(months=Months_Count-1)

    L1=0;L2=0
    for i in range(len(nc_date_str)):

        if(date1==nc_date_str[i]):
            L1=i
            print('L1=',i)
            #print(i,nc_date_str[i],date1,date2)
        if(date2==nc_date_str[i]):
            L2=i
            print('L2=',i)
            #print(i,nc_date_str[i],date1,date2)

    print(L1,L2)
    print(date1,date2)

    Field1 = hgt[L1:L2+1,:,:]
    Field1 = np.mean(Field1,axis=0)
    print(Field1.shape)

    return Field1,lat,lon



#----------------------------------------------------------------------------------------------
def read_climate_model_from_path(ModelPath,ModelPatten,Mon_Init,Year1,Year2,Mon_Pred,MonthCount,var_name1,Check_Year=False):
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

#----------------------------------------------------------------------------------------------
def Process_Model_Dat(filename):

    print(len(filename))
    print(filename)


    type = sys.getfilesystemencoding()
    print(type)

    #sys.exit(0)
    import modes12 as modes12
    #config = ConfigObj(filename,encoding='utf-8')
    #读取配置文件
    #print(config['INFO']['ObsType'])

    INFO = modes12.read_MODES_XML_CFG(filename)

    if(INFO['ModelName']=='Ensemble'):
        #如果类型为集合预报则不处理要素场数据
        return
    ModelPath = INFO['ModelPath']
    ModelPatten = INFO['ModelPatten']
    ModelInitDate = INFO['ModelInitDate']
    ModelPredDate = INFO['ModelPredDate']
    Hindcast_Year_Start = INFO['Hindcast_Year_Start']#Hindcast_Year_Start
    MonthCount = INFO['MonthCount']

    print(ModelInitDate)
    print(Hindcast_Year_Start)
    Year1 = int(Hindcast_Year_Start)
    Year2 = int(ModelInitDate[0:4])+2
    print(Year2)
    Mon_Init = int(ModelInitDate[4:6])
    Mon_Pred = int(ModelPredDate[4:6])  #读取的月值是预报的月份
    #var_name1 = 'z500'
    #return
    Nc_var_list = Get_Nc_VarName_List(ModelPath,ModelPatten)

    print(Nc_var_list)
    #return
    #sys.exit(0)
    dict2={}
    VarField,I_Year,lat,lon = read_climate_model_from_path(ModelPath,ModelPatten,Mon_Init,Year1,Year2,Mon_Pred,MonthCount,'z500',Check_Year=True);
    File_temp_Name ='%s_%4d_%4d_Init_M%02d_Pred_M%02d_%d.BLZ'%(ModelPatten,np.min(I_Year),np.max(I_Year),Mon_Init,Mon_Pred,int(MonthCount))
    print(File_temp_Name)
    #sys.exit(0)
    #return

    #path1 = df.cur_file_dir()

    SKYCLIM1_HOME = os.environ.get('SKYCLIM1_HOME')
    #config['DATA']={}
    File_temp_Name=os.path.join(SKYCLIM1_HOME,'tmp', File_temp_Name)

    #########################################################
    import xml.etree.ElementTree as etree
    tree = etree.parse(filename)
    root = tree.getroot()

    if( root("MODEL_BLZ")>0 ):
        pass
    else:
        etree.SubElement(root,'MODEL_BLZ').text=File_temp_Name



    #root.append(item)
    #indent(root)
    #tree.write('a1.xml',encoding='utf-8',xml_declaration=True)
    tree.write(filename,encoding='utf-8',xml_declaration=True)
    #########################################################

    #config['INFO']['MODEL_BLZ'] = File_temp_Name
    #config.write()

    print(I_Year)
    #sys.exit(0)
    if(not os.path.isfile(File_temp_Name)):
    #if(1):
        for varname in Nc_var_list:

            VarField,I_Year,lat,lon = read_climate_model_from_path(ModelPath,ModelPatten,Mon_Init,Year1,Year2,Mon_Pred,MonthCount,varname)
            dict2[varname]=VarField
            dict2['I_Year']=I_Year
            dict2['lat']=lat
            dict2['lon']=lon


        #df.save_obj(dict2,'hgt_all.npy')

        df.save_obj_blosc(dict2,File_temp_Name)

        #df.get_sel_list_from_all_list()

    #要做一个工作，挑选最近的月份的文件名
        #1、挑选指定预报月份的最后的一个文件名的日期
        #2、按年循环，提取指定年份，2月的所有的文件名列表
        #3、提取与之时间距离最近的文件列表
#dict1[i_Y]=

def main():
    df.mytic()
    #filename=r'E:\Projects11\1105_ClimOperation\00_setup_climOperation\Products\201402\201403\ZQ34Z\PAP\NCC-CGCM\201402.201403.ZQ34Z.PAP.NCC-CGCM.BP-CCA_H500_EA\201402.201403.ZQ34Z.PAP.NCC-CGCM.BP-CCA_H500_EA.INI'
    #filename=r'E:\Projects11\1105_ClimOperation\00_setup_climOperation\products\201402\201403\ZQ34Z\PAP\TCC-MRICGCM\201402.201403.ZQ34Z.PAP.TCC-MRICGCM.BP-CCA_H500_EA\201402.201403.ZQ34Z.PAP.TCC-MRICGCM.BP-CCA_H500_EA.INI'
    #filename=r'E:\Projects11\1105_ClimOperation\00_setup_climOperation\products\201402\201403\ZQ34Z\PAP\NCEP-CFS\201402.201403.ZQ34Z.PAP.NCEP-CFS.BP-CCA_H500_EA\201402.201403.ZQ34Z.PAP.NCEP-CFS.BP-CCA_H500_EA.ini'
    SKYCLIM1_HOME = os.environ.get('SKYCLIM1_HOME')

    fn1 = os.path.join(SKYCLIM1_HOME, 'tmp', 'MODES_FILE_LIST.TXT');
    list1  = d_std_lib.readfile_to_list(fn1)

    # for filename in list1:
    #     list2  = d_std_lib.readfile_to_list(filename)
    #     for line2 in list2:
    #         print(line2)
    #     print('*'*79)

    for filename in list1:
        #print(filename)
        df.mytic()
        Process_Model_Dat(filename)
        df.mytoc('cc')

    df.mytoc('总时间消耗 :')

if __name__ == '__main__':
    main()



