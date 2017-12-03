# -*- coding: cp936 -*-
from __future__ import print_function
from __future__ import division
import dfunc as df
import netCDF4 as nc4
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import sys,os
import d_std_lib as d_std_lib
import dclimate as dclim

########################################################
#读取模式产品的配置信息
########################################################
def read_MODES_XML_CFG(IniFileName):
    '''
    #读取模式产品的配置信息
    将XML文件配置文件的信息变为 字典
    兼容gb2312和UTF-8格式，主要应用于UTF-8格式
    '''
    import xml.etree.ElementTree as etree
    tree = etree.parse(IniFileName)
    root = tree.getroot()
    a={}
    for x in root:
        #for y in x:
        #    print(y.attrib["src"]) #这样能把photo节点下的src属性的值全部输出
        b={}
        b[x.tag]=x.text
        a=dict(a.items()+b.items())
        #print(x.tag,x.text.encode('gb2312'),type(x.text))#.encode('utf-8').decode('gb2312'))
        #print(x.text.encode('gb2312'))
    #print(a)
    #df.disp_dict_info(a)
    return a

#----------------------------------------------------------------------------
#读取模式数据
#----------------------------------------------------------------------------
def read_climate_model_from_path(ModelPath,ModelPatten,Mon_Init,Year1,Year2, \
                                 Mon_Pred,MonthCount,var_name1,Only_Check_Year=False):
    ######################################
    #获取文件所有列表
    #Only_Check_Year 代表仅仅只是查找年份的意思
    #当其为True时间，紧紧检查时间，其他为为空

    AllList1 = df.get_dir_all_file_list(ModelPath)
    #print(AllList1)
    #for Line1 in AllList1:
    #    print(Line1)

    re_str = ModelPatten

    # print(re_str)

    #sys.exit(0)
    filelist1 = df.get_sel_list_from_all_list(AllList1,re_str)
    #print('1'*79)
    #for Line1 in filelist1:
    #    print(Line1)
    #print('1'*79)
    #print(filelist1)
    dict1 = {}
    List_Year_Model=[]
    Field_All=[]
    for i_Y in range(Year1,Year2):
        re_str = ModelPatten+'_'+'%4d'%i_Y+'%02d'%Mon_Init

        ### print(i_Y,re_str)
        filelist1 = df.get_sel_list_from_all_list(AllList1,re_str)
        ### print('----step2-----')
        ### print(filelist1)
        #print('22222222222222222222222')

        #print(len(filelist1))
        if(len(filelist1)>0):
            #sys.exit(0)
            #print(filelist1[0])
            List_Year_Model.append(i_Y)
            if(not Only_Check_Year):
                filename1=filelist1[-1] #读取最后的一个文件用于累加做最新的文件数据
                ### print('Selected FileName =',filename1)
                Field1,lat,lon = read_climate_model_dat_One(filename1,i_Y,int(Mon_Pred),int(MonthCount),var_name=var_name1)

                ### print(Field1.shape,i_Y)
                dict1[i_Y]=Field1

    if(Only_Check_Year):
        lat=None;lon=None
        return Field_All,List_Year_Model,lat,lon

    ### print(List_Year_Model)

    shape1 = np.shape(dict1[List_Year_Model[0]])
    Field_All=np.zeros( (len(List_Year_Model),shape1[0],shape1[1] ))

    print(shape1)
    for ii in range(len(List_Year_Model)):
        Tmp_Year = List_Year_Model[ii]
        Field_All[ii,:,:]=dict1[Tmp_Year]

    return Field_All,np.array(List_Year_Model),lat,lon



######################################################
#单个变量的数据
######################################################
def read_climate_model_dat_One(FieldFileName,Specify_Year,Mon,Months_Count,var_name='h500'):
    print(FieldFileName,end='|')
    rootgrp = nc4.Dataset(FieldFileName,'r')
    print(rootgrp.file_format,end='|')
    #print(rootgrp.variables)

    #sys.exit(0)
    #print('----Start read model data---- '*2)
    #sys.exit(0)
    lat = rootgrp.variables['lat'][:];
    lon = rootgrp.variables['lon'][:];
    #level = rootgrp.variables['level'][:];

    if 'level' in rootgrp.variables:
        level = rootgrp.variables['level'][:]
        dinfo["level"]=int(level[ilev])
    else:
        #print("The NC file Can't find level variable",end='')
        print("The NcFile is single Level Nc File ",end='|')

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

    print('Start index=',L1,'End index=',L2 ,end='|')
    print('date1=',date1,'date2=',date2)

    Field1 = hgt[L1:L2+1,:,:]
    Field1 = np.mean(Field1,axis=0)
    #print(Field1.shape)

    return Field1,lat,lon

#############################################################
#计算多年均值
#############################################################
def Proc_Obs_File_To_Anomaly(ObsFileName):
    df_Obs,df_Sta_Info= Process_Obs_File(ObsFileName)

    # index1 = df_Obs.columns
    # print(index1)
    # index1 = index1 >= 1981
    # print(index1)
    # df_Obs_tmp = df_Obs.loc[:,index1]# df_Obs.columns<=2010]
    # index1 = df_Obs_tmp.columns
    # index1 = index1 <= 2010
    # print(index1)
    index1 = np.arange(1981,2011)
    df_Obs_tmp = df_Obs.loc[:,index1]

    df_Obs_mean = df_Obs_tmp.mean(axis=1)
    df_Obs = df_Obs.sub(df_Obs_mean,axis=0)

    return df_Obs,df_Sta_Info,df_Obs_mean

def Proc_Type2_Value_To_Anomaly(I_Year_Obs,Region,StaLatLon):
    I_Year2=np.arange(1981,2011)
    print(I_Year2)
    i_sel1 = np.in1d(I_Year_Obs,I_Year2)
    print(i_sel1)
    Re1 = Region[:,i_sel1]

    #print(Re1.shape)
    Re1 = np.mean(Re1,axis=1)
    Region_Mean=Re1
    np.savetxt('Obs_Mean.txt',np.vstack((StaLatLon.T,Re1)).T,fmt='%7.2f')
    print(Re1)
    print(Region.shape)
    ##################################################
    Re2 = np.tile(Re1,Region.shape[1])
    Re2 = Re2.reshape((-1,Region.shape[0]))
    #print(Re2)
    #sys.exit(0)
    Re2 = Re2.T
    ##################################################
    print(Re2.shape)
    # print(Re2.shape,Region.shape)
    # print(Re2)
    # print(Re1.T)
    Region =Region-Re2
    np.savetxt('Obs_Anomaly.txt',Region,fmt='%6.2f')
    #sys.exit(0)
    return Region,Region_Mean


######################################################################
#2014-06-15用于处理删除常量行的数据
#后期将处理删除 常量个数比例较高的行和列
######################################################################
def del_con_row_file1_file2(file1,file2):
    RegionR = np.genfromtxt(file1)

    # df1 = pd.DataFrame(RegionR[1:,3:],index=RegionR[1:,0],columns=RegionR[0,3:])
    # df1_std = df1.std(axis=1)
    # print(df1.shape)
    # df1  =df1.loc[df1_std!=0,:]
    #
    # print(df1.shape)

    dat1 =RegionR[:,3:]
    idx = np.std(dat1,axis=1)!=0
    #print( idx )
    dat2 = RegionR[idx,:]
    #print(dat2)
    np.savetxt(file2,dat2,fmt='%10.2f')
    return file2

######################################################################
#2014-06-20规范处理观测数据的读取
######################################################################
def Process_Obs_File(ObsFileName):
    print(ObsFileName)
    ObsFileName= del_con_row_file1_file2(ObsFileName,'obs_tmp.txt')
    print(ObsFileName)

    ######################读取观测数据##########################
    RegionR = np.loadtxt(ObsFileName)
    Region = RegionR[1:,3:]
    Stationid = RegionR[1:,0].astype(int)

    I_Year_Obs = RegionR[0,3:]
    print('OBS Year=',I_Year_Obs,'OBS Year Count=',I_Year_Obs.shape)
    df_Obs = pd.DataFrame(Region,index=Stationid,columns=I_Year_Obs.astype(int))
    df_Sta_Info = pd.DataFrame(RegionR[1:,0:3],index=Stationid,columns=['id','lon','lat'])
    return df_Obs,df_Sta_Info

######################################################################
#2014-06-28 预报和实况空间分布图
######################################################################
def Draw_Obs_And_HindCast_Surface(INFO,MODES_Dict,df_Obs,Hindcast,df_Sta_Info,FileName_Pre):
    SKYCLIM1_HOME = os.environ.get('SKYCLIM1_HOME')
    ################################################################
    ObsFileName=INFO['ObsFileName']
    ModelPredDate=INFO['ModelPredDate']
    SYear_For_Pred = ModelPredDate[0:4]  #要预报的年份
    SMon_For_Pred = ModelPredDate[4:6]   #要预报的启始月
    MonthCount=INFO['MonthCount']        #累计的月数

    RegionID=INFO['RegionID']            #区域名称及构建的值
    RegionID=RegionID.encode('gb2312')

    RegionName=INFO['RegionName']        #区域名称
    ObsType=INFO['ObsType']              #观测类型
    LevFile1=INFO['LevFile']             #分级对应的观测文件
    SearchPath =INFO['SearchPath']       #分级对应的观测文件
    KeyWord =INFO['KeyWord']             #分级对应的观测文件
    PredObj_DataType=INFO['DataType']

    Valid_Start_Year=MODES_Dict['VALID_START_YEAR']

    sky_region_config1 = os.path.join(SKYCLIM1_HOME,'config','sky_region_config.xml')

    sShowInfo1=[0,1,1]

    if(INFO['bShowNumber']=='1'):
        sShowInfo1[1]=1
    else:
        sShowInfo1[1]=0
        #####################################################################


    if('True'==MODES_Dict['DRAW_REPREDICTION']):#绘制历史回报值

        df_Obs5 = df_Obs.loc[:,df_Obs.columns>=Valid_Start_Year]
        df_Hindcast5 = Hindcast.loc[:,df_Obs5.columns&Hindcast.columns]
        #df_Field4 = df_Field3.loc[:,df_Obs3.columns<Start_Year]

        for Year1 in df_Obs5.columns:

            Title2 = RegionName.encode('gb2312')+d_std_lib.Get_Pred_Date_String('%d'%Year1,SMon_For_Pred,MonthCount)
            Title2 = Title2+ObsType.encode('gb2312')+'观测'

            if('2'==PredObj_DataType):
                Title2 = Title2+'(距平值)'

            print('redraw=',Year1)
            Obs1 = df_Obs.loc[:,[Year1]]
            df_X_Obs = pd.concat([df_Sta_Info,Obs1],axis=1)
            FileName_Pre2 =FileName_Pre.replace('PRED','%d'%Year1)
            #print(df_X_Obs.iloc[:,1:].values)
            #sys.exit(0)
            if(not os.path.isfile( FileName_Pre2)):
                d_std_lib.draw_Region_Map(sky_region_config1,RegionID,df_X_Obs.iloc[:,1:].values,Levfile=LevFile1,Title=Title2, \
                                          imgfile2=FileName_Pre2,sShowInfo=sShowInfo1)

            #dfunc.PicBorderColor(FileName_Pre2,COLOR='#0000FF',width=1) #加灰色的边框
            print(df_X_Obs)
            #print('-'*40)
        #print(df_Obs5)

        for Year1 in df_Hindcast5.columns:
            print('redraw=',Year1)

            Title2 = RegionName.encode('gb2312')+d_std_lib.Get_Pred_Date_String('%d'%Year1,SMon_For_Pred,MonthCount)
            Title2 = Title2+ObsType.encode('gb2312')+'历史回报'
            Title2 = Title2+'\n由'+INFO['ModelName'].encode('gb2312')+ \
                     '('+INFO['ModelInitDate'].encode('gb2312')+')' \
                     + INFO['DownScaleModelName'].encode('gb2312')+'方法'

            if('2'==PredObj_DataType):
                Title2 = Title2+'(距平值)'

            Obs1 = df_Obs.loc[:,[Year1]]
            df_X_Obs = pd.concat([df_Sta_Info,Obs1],axis=1)

            df_Tmp1 = df_Hindcast5.loc[:,[Year1]]
            df_X_Hindcast = pd.concat([df_Sta_Info,df_Tmp1],axis=1)

            PS,SC,ACC,RMSE=dclim.do_PS(df_X_Obs.iloc[:,-1].values,df_X_Hindcast.iloc[:,-1],20,50)
            print(PS,SC,ACC,RMSE)
            Title2 = Title2+'\n'+'Sc=%4.2f,Acc=%4.2f,RMSE=%5.1f'%(SC,ACC,RMSE)

            #sys.exit('SC')
            #print(df_X_Hindcast)
            FileName_Pre2 =FileName_Pre.replace('PRED','RePred_%d'%Year1)
            if(not os.path.isfile( FileName_Pre2)):
                d_std_lib.draw_Region_Map(sky_region_config1,RegionID,df_X_Hindcast.iloc[:,1:].values,Levfile=LevFile1,Title=Title2, \
                                          imgfile2=FileName_Pre2,sShowInfo=sShowInfo1)