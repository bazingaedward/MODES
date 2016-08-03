# -*- coding: cp936 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
import time as getsystime
from mpl_toolkits.basemap import Basemap,shapefile
import matplotlib.pyplot as plt
import netCDF4 as nc4
from dateutil.relativedelta import relativedelta
import sys,time
#
from dfunc import mytic,mytoc,save_obj,load_obj
import dplotlib as dplot
import dclimate as dclim
import dfunc as df
from  datetime import datetime
###########################

class climate_modes():
    '''

    '''
    def __init__(self,**arg):
        """
         类实体初始化函数,将可能用到的数据实体化
        """
        print('Init climate_diagnosis')

        self.debug=1
        self.I_Year = arg.pop('I_Year',None)  #输入起始年份序列
        self.I_Year_Show= arg.pop('I_Year_Show',self.I_Year[-1]);  #需要单独显示的年份

        print('Years=',self.I_Year)
        print('Show&Pred Year=',self.I_Year_Show)
        
        self.Mon          = None          #月份
        self.Months_Count = None #月份数

        self.I_Year_Pred = None  #预测的年份
        self.Field_Pred  = None   #预测单点场资料
        self.Field_Pred_Info = None #模式资料信息

        self.I_Year_Reanalysis = None #再分析资料起始年数
        self.Field_Reanalysis  = None #再分析资料的场资料

        self.I_Year_Region = None
        self.Region=None

        self.lons = np.arange(0, 360, 2.5, dtype=float)
        self.lats = np.arange(90, -90 - 1, -2.5, dtype=float)
        #ObjName 为预报对象名称
        self.ObjName = ''
        #读取的高度场信息
        self.Model_Field_info ={}

    def Init_Parameter(self,Model_info,Mon,Months_Count):
        '''
            初始化预报月和时段参数，即到底是月还是季
        '''
        self.Model_info = Model_info
        self.Mon = Mon
        self.Months_Count = Months_Count

    def Init_Sta_Dat(self,FileName):
        '''
            初始化站点数据
        '''
        RegionR = np.loadtxt(FileName)
        self.Region_LeftHead = RegionR[:,0:3]
        self.Region = RegionR[1:,3:]
        self.I_Year_Region = RegionR[0,3:]
        self.StaLatLon = RegionR[1:,0:3]
        print(self.Region.dtype)
        #if(self.debug):
        #    print(self.Region,self.I_Year_Region)

    #获取模式的初始化信息
    def Get_Model_Info(self,FileName,**arg):
        rootgrp = nc4.Dataset(FileName,'r')
        print(rootgrp.file_format)
        a={}
        for name in rootgrp.ncattrs():
            a[name]= getattr(rootgrp,name)
            print('Global attr',name, '=', getattr(rootgrp,name))
        print(a)
        self.Model_Field_info=a

    def Init_Model_Field_Dat(self,FileName,**arg):
        '''

        '''
        self.model_var_name = arg['var_name']
        print(self.model_var_name)
        #sys.exit(0)
        self.Field_Pred,self.I_Year_Pred = read_climate_model_dat(\
                                   FileName,self.I_Year,self.Mon,self.Months_Count,**arg)

    def Init_Reanalysis_Field_Dat(self,FileName,Offset=0,**arg):
        '''
        从netcdf文件中读取NCEP再分析资料数据
        '''
        if None==self.Mon or None==self.Months_Count:
            print('Error Please Init Month value by "Init_Parameter(Mon,Months_Count)" Method')
            sys.exit(0)
        from dfunc import Read_Ncep_Hgt2
        self.Field_Reanalysis,self.I_Year_Reanalysis,self.Fieldinfo  = Read_Ncep_Hgt2(FileName,I_Year=self.I_Year,\
                                             Mon=self.Mon,Months_Count=self.Months_Count,FieldOffset=Offset,
                                             **arg)

    def Align_Region_and_Field(self,latmin=-90,latmax=90,lonmin=0,lonmax=360):
        '''
        预报模块，需要进一步完善，有很多内容需要进一步深入
        '''
        I_Year_Region = self.I_Year_Region
        I_Year_Pred = self.I_Year_Pred

        print('站点的年份  =',I_Year_Region)
        print('预报场的年份=',I_Year_Pred)


        #print(self.Field[:,0,0])
        #print(self.FieldP[:,0,0])
        I_Year_Pred = np.array(I_Year_Pred)
        print('type',type(I_Year_Pred),type(I_Year_Region))


        Region = self.Region[:,np.in1d(I_Year_Region,I_Year_Pred)]
        I_Year_Region=I_Year_Region[np.in1d(I_Year_Region,I_Year_Pred)]
        print("与预报场对齐后的站点年份 I_Year_Region =\n",I_Year_Region)
        print("预报场年份 I_Year_Pred =\n",I_Year_Pred)

        I_Year_Pred_Not_in_Region = I_Year_Pred[np.logical_not(np.in1d(I_Year_Pred,I_Year_Region)) ]
        #sys.exit(0)
        print("不再观测资料中的年份 I_Year_Pred_Not_in_Region =\n", I_Year_Pred_Not_in_Region)
        print("选定的预报年 I_Year_Show = \n",self.I_Year_Show)

        Field_Pred = self.Field_Pred

        print('PredShape=',np.shape(Field_Pred))
        print('选取预报场中的东亚区域%d~%d,%d~%d'%(latmin,latmax,lonmin,lonmax))
        ##################################################

        lat1 = np.where(self.lats <= latmax,True,False) #45
        lat2 = np.where(self.lats >= latmin,True,False) #0

        Select_lat = np.logical_and(lat1,lat2)

        lon1 = np.where(self.lons <= lonmax,True,False) #135
        lon2 = np.where(self.lons >= lonmin,True,False) #70
        Select_lon = np.logical_and(lon1,lon2)
        print('Select_lat Shape=',Select_lat.shape,'Select_lon Shape=',Select_lon.shape)

        Field2 = Field_Pred[:,:,Select_lon]
        Field2 = Field2[:,Select_lat,:]
        n1 = Field2.shape
        print('Selected Area Array Shape=',np.shape(Field2),n1[0])
        Field2 = np.reshape(Field2,(n1[0],-1))
        print('Selected Area Array Shape=',np.shape(Field2))
        #挑选在
        Field_Hindcast = Field2[np.in1d(I_Year_Pred,I_Year_Region),:].T #注意转置了

        print('Region Data Shape    = ',Region.shape)
        print('Field_Hindcast Shape = ',Field_Hindcast.shape)

        Field_Select4Pred = Field2[np.in1d(I_Year_Pred,self.I_Year_Show),:]
        print('Field_Select4Pred Shape = ',Field_Select4Pred.T.shape)


        #FieldP = self.FieldP[:,self.p_np3]  #等于过滤后的场文件
        #FieldP = FieldP.T
        #FieldP2 = FieldP[:,np.in1d(I_YearP,I_Year)]

        #print(FieldP2.shape,np.atleast_2d(FieldP[:,-1]).T.shape)
        #print('FieldP.shape = ',FieldP.shape)
        #print('FieldP2.shape = ',FieldP2.shape)
        #print('Region.shape = ',Region.shape)
        #sys.exit(0)

        PredDict={}
        PredDict['StaLatLon'] = self.StaLatLon
        PredDict['Region']=Region
        PredDict['I_Year_Region']=I_Year_Region
        PredDict['I_Year_Pred']=self.I_Year_Show

        PredDict['Field_Hindcast']=Field_Hindcast
        PredDict['Field_Select4Pred']=Field_Select4Pred.T
        return(PredDict)

    def Pred_EOF_CCA(self,PredDict,OutPreFile='out.txt'):
        Field_Hindcast    = PredDict['Field_Hindcast']
        Region            = PredDict['Region']
        Field_Select4Pred = PredDict['Field_Select4Pred']
        print(np.shape(Field_Select4Pred))


        self.X_Pre = dclim.dpre_eof_cca(Field_Hindcast,Region,np.atleast_2d(Field_Select4Pred),4)
        #sys.exit(0)
        PredDict['Pred']=self.X_Pre
        #sys.exit(0)
        print(self.X_Pre.shape)
        self.out = np.hstack((self.StaLatLon,self.X_Pre))
        PredDict['Pred_Out']=self.out
        print('Pred Year is ',self.I_Year_Show)
        np.savetxt(OutPreFile,self.out,fmt='%5d %7.2f %7.2f %7.2f',delimiter=' ')
        return PredDict


    def Pred_EOF_CCA_Validation(self,PredDict,bShowImage=0,Title='',FileStrHead=''):
        '''
        交叉验证BPCCA模块，需要进一步完善，有很多内容需要进一步深入
        '''

        I_Year = PredDict['I_Year_Region']
        I_YearP = PredDict['I_Year_Pred']

        Region = PredDict['Region']
        Region_XPre=np.zeros_like(Region)

        FieldP2 = PredDict['Field_Hindcast']


        print('FieldP2.shape = ',FieldP2.shape)
        print('Region.shape = ',Region.shape)

        shape1 = FieldP2.shape

        print(type(I_Year),type(I_YearP))
        #print(I_Year.shape,I_YearP.shape)
        #return

        PS= np.zeros((shape1[1]))
        SC= np.zeros((shape1[1]))
        ACC= np.zeros((shape1[1]))
        RMSE= np.zeros((shape1[1]))

        for i in range(shape1[1]):
            #continue
            FieldP3 = FieldP2
            FieldP2_H = np.delete(FieldP3,i,axis=1)
            FieldP2_P = FieldP2[:,i]

            Region2 = Region
            Region_P = Region2[:,i]
            Region_H = np.delete(Region2,i,axis=1)

            #print(FieldP2_H.shape,Region_H.shape)
            print('CCA Vali','%02d-Year=%04d'%(i,I_Year[i]),end=' ')
            X_Pre = dclim.dpre_eof_cca(FieldP2_H,Region_H,np.atleast_2d(FieldP2_P).T,4)
            print(Region_XPre.shape,X_Pre.shape)
            Region_XPre[:,i]= X_Pre[:,0]

            PS[i],SC[i],ACC[i],RMSE[i]=dclim.do_PS(X_Pre,Region_P,L1=20,L2=50)
            print('%5.2f,%5.2f,%5.2f,%5.2f'%(PS[i],SC[i],ACC[i],RMSE[i]))

        fig=plt.figure()
        #plt.plot(I_Year,SA,'-o', ms=5, lw=1, alpha=0.7, mfc='blue')

        plt.plot(I_Year,PS/100.0,'-*', ms=4, lw=2, alpha=0.7, mfc='blue')
        plt.plot(I_Year,SC,'-^', ms=4, lw=1.5, alpha=0.7, mfc='green')
        plt.bar(I_Year,ACC,color='blue',width=0.5)

        plt.xlim(I_Year[0]-1,I_Year[-1]+1)

        cnfont = dclim.GetCnFont()
        ptitle = u'交叉检验 AVG PS=%5.2f,SC=%5.2f,ACC=%5.2f'%(np.mean(PS)/100.0,np.mean(SC),np.mean(ACC))
        ptitle = Title.decode('gb2312')+"\n"+ptitle
        plt.title(ptitle,fontproperties=cnfont)

        #ax=fig.add_subplot(111)
        plt.legend((u'PS/100','SC',u'ACC',),0,prop=cnfont)

        #print(np.sign(Region))
        #print(np.sign(Region_XPre))
        
        All_Score = np.vstack((I_Year,PS,SC,ACC))
        FileName0 = FileStrHead+'Vali.txt'

        np.savetxt(FileName0,All_Score,fmt='%7.2f')

        #np.savetxt
        #调试输出信息
        if(0):
            print(All_Score)
            print('sleep...!')
            import time
            for ii in range(50):
                print(ii,end=' ')
                time.sleep(1)

        plt.grid()
        FileName1 = FileStrHead+'Vali.png'
        plt.savefig(FileName1)
        df.CutPicWhiteBorder('Vali.png')
        if(bShowImage):
            plt.show()

        
        Region_XPre2=np.vstack((I_Year,Region_XPre))
        Region_XPre2=np.hstack((self.Region_LeftHead,Region_XPre2))
        np.savetxt(FileStrHead+'XPre.txt',Region_XPre2,fmt='%9.2f')
        return Region_XPre

    def Calc_Region_SameRate(self,Region,Region_XPre,StaLatLon):
        ###########################################################
        Region_Sign = np.sign(Region)*np.sign(Region_XPre)
        Region_Sign2 =  np.where(Region_Sign>0,Region_Sign,0)
        print(Region_Sign2)
        X_Pre_Sign = np.mean(Region_Sign2,axis=1)
        print(StaLatLon.shape,X_Pre_Sign.shape)

        X_Pre_Sign_out= np.hstack((StaLatLon,np.atleast_2d(X_Pre_Sign).T))
        np.savetxt('sign.txt',X_Pre_Sign_out,fmt='%5d %7.2f %7.2f %7.2f',delimiter=' ')
        ##################################################################

    def Corr_Field_Reanalysis_and_Field_Pred(self):
        '''
        高度场和预报场相关
        '''
        print('*'*80)
        I_Year_Reanalysis = self.I_Year_Reanalysis
        I_Year_Pred = self.I_Year_Pred
        print("NCEP Years\n",I_Year_Reanalysis)
        print("Model Years\n",I_Year_Pred)

        print(np.shape(self.Field_Reanalysis),"\n",np.shape(self.Field_Pred));
        print("NCEP Date [0,0]: \n",self.Field_Reanalysis[:,0,0])
        print("Model Date [0,0]:\n",self.Field_Pred[:,0,0])

        #sys.exit(0)

        FieldN = self.Field_Reanalysis[np.in1d(I_Year_Reanalysis,I_Year_Pred),:,:]
        FieldP = self.Field_Pred[np.in1d(I_Year_Pred,I_Year_Reanalysis) ,:,:]

        print(self.Field_Reanalysis.shape)
        print(FieldN.shape)
        print(self.Field_Pred.shape)
        print(FieldP.shape)
        m=FieldP.shape

        FieldN = FieldN.reshape(m[0],-1)
        FieldP = FieldP.reshape(m[0],-1)

        r_np=np.zeros(m[1]*m[2])
        p_np=np.zeros(m[1]*m[2]) #73*144
        
        from scipy.stats import pearsonr
        for i in range(m[1]*m[2]):
            r_np[i],p_np[i] = pearsonr(FieldN[:,i],FieldP[:,i])

        self.r_np = r_np.reshape((m[1],m[2]))
        self.p_np = p_np.reshape((m[1],m[2]))



    def Draw_Cross_Corr(self,ptitle='',func='ncep2sta',showimg=0):
        '''
        绘制交叉检验函数的函数
        '''

        from dateutil.relativedelta import relativedelta

        datestr1 ='%04d-%02d-01'%(self.I_Year_Pred[-1],self.Mon)
        date1 = datetime.strptime(datestr1,'%Y-%m-%d')
        date2 = date1+relativedelta(months=self.Months_Count-1)

        if(self.Months_Count>1):
            Title_DateStr1 = datetime.strftime(date1,' %b') #+'-'+datetime.strftime(date2,'%b')
            
            Title_DateStr2 = datetime.strftime(date1,' %Y %b')
            for ii in range(self.Months_Count -1):
                #Title_DateStr2 = Title_DateStr2+'-'+ datetime.strftime(date1+relativedelta(months=self.Months_Count-1-ii),'%b')
                Title_DateStr2 = Title_DateStr2+'-'+ datetime.strftime(date1+relativedelta(months=ii+1),'%b')
                #Title_DateStr1 = Title_DateStr1+'-'+ datetime.strftime(date1+relativedelta(months=self.Months_Count-1-ii),'%b')
                Title_DateStr1 = Title_DateStr1+'-'+ datetime.strftime(date1+relativedelta(months=ii+1),'%b')

        else:
            Title_DateStr1 = datetime.strftime(date1,' %b')
            Title_DateStr2 = datetime.strftime(date1,' %Y-%b')


        imshow=showimg
        if 'sta_ncep_cross_corr'==func or 1==func:
            print(self.r2)
            dplot.drawhigh4corr(self.r2,self.lons,self.lats,ptype=1,\
                                ptitle=ptitle+u'站点和再分析交叉分析图 sta_ncep_cross_corr'+Title_DateStr1,\
                                imgfile='011SNCC.png',showimg=imshow)

        if 'sta_ncep_cross_corr_pval'==func or 2==func:
            dplot.drawhigh4corr(self.p2,self.lons,self.lats,ptype=1,\
                                ptitle=ptitle+u'站点和再分析交叉分析信度 sta_ncep_cross_corr_pval'+Title_DateStr1,\
                                imgfile='012SNCCP.png',showimg=imshow)

        if 'pred_ncep_corr'==func or 3==func:  #模式预测结果与再分析资料相关图

            ptitle1=self.Model_info + ' Init(' + self.Model_Field_info['Init Time']+')' +" Forecast "\
                   + self.model_var_name +" \n "+Title_DateStr1

            ptitle1=u'模式预测结果与再分析资料相关图 '+ " \n "+ptitle1

            dplot.drawhigh4corr(self.r_np,self.lons,self.lats,ptype=1,\
                                ptitle=ptitle1,
                                imgfile='013PreNC.png',showimg=imshow)
            hgt7 = self.r_np
            hgt7 = np.where(hgt7<0.0,np.nan,hgt7)
            hgt7 = np.where(self.p_np>0.1,np.nan,hgt7)

            dplot.drawhigh4corr(hgt7,self.lons,self.lats,ptype=1,\
                                ptitle=ptitle1,\
                                imgfile='013PreNC_mask2.png',showimg=imshow)

            dplot.drawhigh4corr2(self.r_np,self.p_np,ptype=1,\
                                ptitle=ptitle1,\
                                imgfile='013PreNC_mask.png',showimg=imshow)

        if 'pred_ncep_corr_pval'==func or 4==func:


            ptitle1=self.Model_info + ' Init(' + self.Model_Field_info['Init Time']+')' +" Forecast "\
                   + self.model_var_name +" \n "+Title_DateStr1

            ptitle1=u'模式预测与在分析相关图置信度检验 '+ " \n "+ptitle1

            dplot.drawhigh4corr(self.p_np,self.lons,self.lats,ptype=1,\
                                ptitle=ptitle1,\
                                imgfile='014PreNCP.png',showimg=imshow)

        if  5==func:
            #环流场的实际值，例如500hPa高度场的实际的值
            #FieldP_end =self.FieldP_Filter[0,:,:]

            print(np.in1d(self.I_Year_Pred,self.I_Year_Show))
            Ary_Sel_Year = np.in1d(self.I_Year_Pred,self.I_Year_Show)
            print('&'*80)
            print(np.shape(self.I_Year_Pred),np.shape(self.I_Year_Show))

            #FieldP_end =self.FieldP[-2,:,:]
            FieldP_end =self.Field_Pred[Ary_Sel_Year,:,:]
            FieldP_end =FieldP_end[0,:,:]

            #print(np.shape(FieldP_end[0,:,:]))
            #sys.exit(0)

            #print(FieldP_end)
            #np.reshape(self.FieldP[0,:,:],(73,144))
            #initial time
            print(self.Model_Field_info['Init Time'])
            ptitle=self.Model_info + ' Init(' + self.Model_Field_info['Init Time']+')' +" Forecast "\
                   + self.model_var_name +" \n "+'%04d'%self.I_Year_Show +' '+ Title_DateStr1
            dplot.drawhigh(FieldP_end,self.lons,self.lats,ptype=1,\
                           ptitle=ptitle,\
                                imgfile='015PreNCP_Real.png',showimg=imshow)


        if  6==func:
            #环流场的距平值


            FieldP_avg = np.mean(self.Field_Pred,axis=0)

            #print(np.in1d(self.I_YearP,self.I_Year_Show))
            Ary_Sel_Year = np.in1d(self.I_Year_Pred,self.I_Year_Show)
            #print('&'*80)
            #print(np.shape(self.I_YearP),np.shape(self.I_Year_Show))
            #FieldP_end =self.FieldP[-2,:,:]
            FieldP_end =self.Field_Pred[Ary_Sel_Year,:,:]
            FieldP_end =FieldP_end[0,:,:]
            FieldP_end =FieldP_end - FieldP_avg


            #FieldP_end =self.FieldP[-1,:,:]- FieldP_avg

            #FMax = np.mean(np.abs( FieldP_end.flatten()) ) *2.0
            FMax = np.max(np.abs( FieldP_end.flatten()) )


            #print('aaaaaa',FMax)
            #FMin = np.min(FieldP_end.flatten())
            #print('bbbbbb',FMin)
            #FMax = ( FMax+abs(FMin) )/2.0
            #print('cccccc',FMax)

            import math
            TMax=FMax

            #TMax= math.ceil(FMax/10.0)
            #print('TMax1=',TMax)
            #TMax= TMax*10
            #print('TMax2=',TMax)
            #lev1 = np.linspace(-1*TMax,TMax,11)
            #lev1 = np.linspace(-TMax,TMax,21)  #NCC
            lev1 = get_colormap_level(TMax)
            #cmap_str='RdYlBu' 'bwr'
            #print('lev1=',lev1)
            #sys.exit(0)

            ptitle=self.Model_info + ' Init(' + self.Model_Field_info['Init Time']+')' +" Forecast "\
                   + self.model_var_name +" Anomaly\n "+'%04d'%self.I_Year_Show +' '+ Title_DateStr1
            #ptitle=self.Model_info +' forecast '+ self.model_var_name + ' Anomaly'+'%04d'%self.I_Year_Show +' '+ Title_DateStr1
            dplot.drawhigh(FieldP_end,self.lons,self.lats,ptype=1,\
                           ptitle=ptitle,imgfile='016PreNCP_Anomaly.png',showimg=imshow,cmap_str='seismic',lev=lev1)

        if  7==func:
            #环流场经过过滤的距平值
            FieldP_avg = np.mean(self.FieldP_Filter[0:-1,:,:],axis=0)
            FieldP_end =self.FieldP_Filter[-1,:,:]- FieldP_avg
            dplot.drawhigh(FieldP_end,self.lons,self.lats,ptype=1,\
                           ptitle=ptitle+u'\npred_ncep_corr_pval'+Title_DateStr2,\
                                imgfile='017PreNCP_Anomaly_Filter.png',showimg=imshow)

        if  8==func:
            #画高度场的曲线
            lev1 = np.array([5840,5880])
            FieldP_avg = np.mean(self.FieldP[0:-1,:,:],axis=0)
            FieldP_end =self.FieldP[-1,:,:]
            #黑线为平均值
            #红线为预测值
            ptitle=self.Model_info +' '+ self.model_var_name + ' 588 line avg and forecast'+Title_DateStr2
            dplot.drawhigh5880Line(FieldP_end,FieldP_avg,self.lons,self.lats,ptype=1,\
                           ptitle=ptitle,\
                                imgfile='018PreNCP_588line.png',showimg=imshow,lev=lev1)








def get_colormap_level(TMax):

    a = '%e'%TMax;
    print('TMax=',TMax,'a=',a)

    a1 = a[0:a.index('e')]
    a2 = a[a.index('e')+1:]
    #print(a1,a2,a)

    import math,string

    b1=int(string.atof(a1)*10)

    b1_quotient = b1//10
    b1_mod = b1%10
    print('b1=',b1,'b1_q',b1_quotient,'b1_mod',b1_mod)


    b2=string.atof(a2)
    b3 = 21
    #print(b3)
    TMax = b1_quotient*(10**b2)

    lev1 = np.linspace(-TMax,TMax,num=b3)  #NCC
    #print('lev1=',lev1)
    #sys.exit(0)
    return lev1



#------------------------------------------------------------------------------
def read_climate_model_dat(FieldFileName,I_Year,Mon,Months_Count,var_name='hgt'):
    print(FieldFileName)
    rootgrp = nc4.Dataset(FieldFileName,'r')
    print(rootgrp.file_format)
    #print rootgrp.variables
    print('----Start read model data---- '*4)

    lat = rootgrp.variables['lat'][:];
    lon = rootgrp.variables['lon'][:];
    #level = rootgrp.variables['level'][:];

    if 'level' in rootgrp.variables:
        level = rootgrp.variables['level'][:]
        dinfo["level"]=int(level[ilev])
    else:
        print("Can't find level variable")

    times = rootgrp.variables['time'];
    hgt = rootgrp.variables[var_name];

    #生成netcdf时间维数据
    nc_date_str = nc4.num2date(times[:],units=times.units)
    from  datetime import datetime

    year1 = I_Year[0]
    I_Year2 = np.arange(year1,2050)
    #print(I_Year2)

    for tmpy in I_Year2:
        #print('444=',type(tmpy))
        s1 ='%04d-%02d-01'%(tmpy,Mon)
        date1 = datetime.strptime(s1,'%Y-%m-%d')
        date2 = date1 + relativedelta(months=Months_Count-1)
        if(date2>nc_date_str.max()):
            break
        year2 = tmpy
        #print(date2)

    I_Year2 =  range(int(year1),int(year2)+1) #选定的年份
    I_Year3 = []

    PField = np.zeros( (len(I_Year2),len(lat),len(lon)) )

    for i in range(len(nc_date_str)):
        datestr_c =  datetime.strftime(nc_date_str[i],'%Y-%m')
        #print(datestr_c)
        for tmpy in I_Year2:
            #print(tmpy)
            s1 ='%04d-%02d'%(tmpy,Mon)

            if s1==datestr_c:
                I_Year3.append(tmpy)
                L1 = i
                L2 = L1+Months_Count
                print(s1,'L1=',L1,'L2=',L2,end=' ')
                j=tmpy-I_Year2[0]    #注意错位，非常重要，需要修改此处代码
                print('%02d'%j,end=' ')
                ###Field[j,:,:]=np.mean(hgt[L1:L2,0,:,:],axis=0)
                print('data shape =',np.shape(hgt[L1:L2,:,:]),end='')  #打出读取的数值的维数

                PField[j,:,:]=np.mean(hgt[L1:L2,:,:],axis=0)
                print(np.shape( PField[j,:,:] ))


    print('I_Year_Pred I_Year3=',I_Year3)
    I_Year2 = np.array(I_Year2)
    PField = PField[np.in1d(I_Year2,I_Year3),:,:]

    shape1 = PField.shape
    for j in range(shape1[0]):
        print('mean pField',j,I_Year3[j],end=' ')
        print(np.mean(PField[j,:,:]),end=' ')
        print(PField[j,0,0])
        
        if(0==np.mean(PField[j,:,:])):
            PField[j,:,:]=np.nan
            PField = np.delete(PField,j,axis=0)
            I_Year3 = np.delete(I_Year3,j,axis=0)
            break


    print('len_I_Year=%d,len lat= %d,len lon =%d'%(len(I_Year3),len(lat),len(lon)) )
    print('read model Field Shape = ',PField.shape)
    rootgrp.close()
    print('***End read Mode data***'*5)
    print('注意剔除的年份',I_Year3)
    return PField,I_Year3


###########################################################
#获取供系统运行的各类参数
###########################################################

def getDClimDiagini():
    #config1 = ConfigParser.ConfigParser()
    #config1.readfp(open(FileName))
    from dateutil.relativedelta import relativedelta
    import sys,os
    import time
    import ConfigParser

    MODES_PATH = os.environ.get('MODES')
    #print(MODES_PATH)
    #sys.exit(0)
    if(None==MODES_PATH):
        Ini_File_Name='Modes4Pred.ini'
    else:
        Ini_File_Name=os.path.join( MODES_PATH,'Modes4Pred.ini')

    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(Ini_File_Name))

    dict1 = {}
    ###########NCEP 月份信息################
    dict1['Year'] = int( config1.get("FieldPara","Year") )
    dict1['Month1'] = int( config1.get("FieldPara","Month") )
    dict1['Interval'] = int(config1.get("FieldPara","Interval") )
    dict1['Count'] = int(config1.get("FieldPara","Count") )
    #获取站点文件，异常重要
    dict1['StaDatFile'] = config1.get('FieldPara',"StaDatFile")

    #datestr1 ='%04d-%02d-01'%(self.I_YearP[-1],self.Mon)
    #date1 = datetime.strptime(datestr1,'%Y-%m-%d')
    #date2 = date1+relativedelta(months=self.Months_Count-1)

    date1 =datetime.strptime('%04d-%02d-01'%(dict1['Year'],dict1['Month1']),'%Y-%m-%d')
    print('原始日期',date1)
    date1 = date1 + relativedelta(months =  dict1['Interval'])
    print('累加间隔日期',date1)

    dict1['Year'] = int( datetime.strftime(date1,'%Y') )
    dict1['Month'] = int( datetime.strftime(date1,'%m') )

    ###########Model 输出资料信息#################

    dict1['Model_Name'] = config1.get("FieldPara","ModelName")
    dict1['Model_FileName'] = config1.get("FieldPara","ModelFileName")
    MethodSection = config1.get("FieldPara","MethodSection")
    dict1['Title'] = config1.get('FieldPara',"Title")

    dict1['ModelName']= config1.get('FieldPara',"ModelName")
    dict1['MethodName']= config1.get('FieldPara',"MethodName")
    #dict1['MethodName']=config1.get('FieldPara',"MethodSection")
    print(dict1['MethodName'])
    #sys.exit(0)

    #dict1['FileStrHead']='%s_%s_%d-%02d-%0d-%d'%(dict1['ModelName'],dict1['MethodName'],\
    #     dict1['Year'],dict1['Month'],dict1['Interval'],dict1['Count'])

    ###########NCEP资料信息#################
    #dict1['NCEP_FileName'] = config1.get("FieldPara","NCEP_FileName",'')
    #dict1['NCEP_var_name'] = config1.get("FieldPara","NCEP_var_name")
    #dict1['NCEP_level'] = int(config1.get("FieldPara","NCEP_level"))

    config2 = ConfigParser.ConfigParser()


    if(None==MODES_PATH):
        Ini_File_Name='MODES.ini'
    else:
        Ini_File_Name=os.path.join( MODES_PATH,'MODES.ini')
    config2.readfp(open(Ini_File_Name))
    #sys.exit(0)
    print('MethodSection')
    dict1['NcVarList'] = config2.get(MethodSection,"NcVarList")
    dict1['bShowImage'] = int(config2.get('GLOBAL',"bShowImage"))
    dict1['bDrawNumber'] = int(config2.get('GLOBAL',"bDrawNumber"))
    dict1['bDrawNumber'] = int(config2.get('GLOBAL',"bDrawNumber"))

    #str1 = config2.get('GLOBAL',"MODES_NCEP_REANALYSIS_MONTH_PRESSURE_PATH")
    #dict1['MODES_NCEP_REANALYSIS_MONTH_PRESSURE_PATH']= str1.replace('${INITDIR}',MODES_PATH)
    #dict1['Region_Section']= config2.get('GLOBAL',"Region_Section")

    #dict1['FileStrHead']='%d-%02d-%0d-%d_%s_%s_%s'%(dict1['Year'],dict1['Month1'],dict1['Interval'],dict1['Count'],\
    #                                                    dict1['ModelName'],dict1['MethodName'],dict1['Region_Section'])

    #print(dict1['FileStrHead'])

    #print(dict1)
    #for  i in dict1.keys():
    #    print(i,'=>', dict1[i])
    return dict1


#--------------------------------------------------------------------------------
'''
    def Draw_Cross_Corr_2(self,ptitle='Cross_Corr P-value',func='ncep2sta',showimg=1,title=''):

        #绘制交叉检验函数的函数

        imshow=showimg
        if 'sta_ncep_cross_corr'==func or 1==func:
            dplot.drawhigh4corr(self.r2,self.lons,self.lats,ptype=1,ptitle=title,\
                                imgfile='021SNCC.png',showimg=imshow)

        if 'sta_ncep_cross_corr_pval'==func or 2==func:
            dplot.drawhigh4corr(self.p2,self.lons,self.lats,ptype=1,ptitle=ptitle+'\nsta_ncep_cross_corr_pval',\
                                imgfile='022SNCCP.png',showimg=imshow)

        if 'pred_ncep_corr'==func or 3==func:
            dplot.drawhigh4corr(self.r_np,self.lons,self.lats,ptype=1,ptitle=ptitle+'\npred_ncep_corr',\
                                imgfile='023PreNC.png',showimg=imshow)

        if 'pred_ncep_corr_pval'==func or 4==func:
            dplot.drawhigh4corr(self.p_np,self.lons,self.lats,ptype=1,ptitle=ptitle+'\npred_ncep_corr_pval',\
                                imgfile='024PreNCP.png',showimg=imshow)

        if  5==func:
            FieldP_end =self.FieldP_Filter[0,:,:]
            #np.reshape(self.FieldP[0,:,:],(73,144))
            dplot.drawhigh(FieldP_end,self.lons,self.lats,ptype=1,ptitle=ptitle+'\npred_ncep_corr_pval',\
                                imgfile='025PreNCP_test.png',showimg=imshow)

        if  6==func:
            FieldP_avg = np.mean(self.FieldP[0:-1,:,:],axis=0)
            FieldP_end =self.FieldP[-1,:,:]- FieldP_avg
            dplot.drawhigh(FieldP_end,self.lons,self.lats,ptype=1,ptitle=ptitle+'\npred_ncep_corr_pval',\
                                imgfile='026PreNCP_anomaly.png',showimg=imshow)

        if  7==func:
            FieldP_avg = np.mean(self.FieldP_Filter[0:-1,:,:],axis=0)
            FieldP_end =self.FieldP_Filter[-1,:,:]- FieldP_avg
            dplot.drawhigh(FieldP_end,self.lons,self.lats,ptype=1,ptitle=ptitle+'\npred_ncep_corr_pval',\
                                imgfile='027PreNCP_anomaly_Filter.png',showimg=imshow)
'''
