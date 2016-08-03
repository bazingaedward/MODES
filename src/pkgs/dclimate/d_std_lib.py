# -*- coding: cp936 -*-
from __future__ import print_function
from dateutil.relativedelta import relativedelta
from dfunc import mytic,mytoc
import os,re,time,sys
import numpy as np

from __init__ import Spatial_Data
#------------------------------------------------------------------------------
#获取脚本文件的当前路径
def cur_file_dir():
    #获取脚本路径
    path = sys.path[0]
    #判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

#------------------------------------------------------------------------------
#
#################################################

def move_myfile_to_dir(dir1,restr,dst):
    import shutil
    filelist = get_dir_list( dir1,restr)
    for L1 in filelist:
        print(L1)
        target_file = os.path.join(dst, os.path.basename(L1) )
        if( os.path.isfile( target_file ) ):
            os.remove(target_file)
        shutil.move(L1,dst)
        
#------------------------------------------------------------------------------
#
#################################################
def get_sanxia_date_info():
    ########################################

    date_str = time.strftime('%Y-%m-%d',time.localtime(time.time()-0))
    from  datetime import datetime
    date2 = datetime.strptime(date_str,'%Y-%m-%d')

    #预报下一个月的值，此处month=1 即加一个月
    date2 = date2 + relativedelta(months=1)
    date2_str = datetime.strftime(date2,'%Y-%m-%d')
    ###############################################
    SYear_For_Pred = int( date2_str[0:4])
    pre_smon = int( date2_str[5:7])
    pre_sday = int( date_str[8:10] )

    if(int(pre_sday)<=20):
        pre_sday=9
    else:
        import calendar
        _,days = calendar.monthrange(int(SYear_For_Pred),int(date_str[5:7]))
        print(days)
        #sys.exit(0)
        pre_sday= days-2

    return SYear_For_Pred,pre_smon,pre_sday

def get_dir_list(dirname, re_str):
    '''
    获取目录下的文件，可能不带递归
    #m = re.search(r'hello', 'hello world!')
    #print m

    '''
    #print(dirname)
    flist = []
    for dirpath, dirnames, filenames in os.walk(dirname):
        #print(filenames)
        for filename in filenames:
            #print(filename)
            #if os.path.splitext(filename)[1] == '.grb2':
            '''
            pattern = re.compile(re_str)
            match = pattern.match(filename)
            filepath = os.path.join(dirpath, filename)
            if match:
                flist.append(filepath)
            '''
            if(re.search(re_str, filename)):
                filepath = os.path.join(dirpath, filename)
                flist.append(filepath)
    flist.sort()
    return flist

#------------------------------------------------------------------------------
#迭代器读取文件
#------------------------------------------------------------------------------
def readfile_to_list(filename):
    '''
    读取文件内容到List再将其中的内容取掉\n
    返回list
    '''
    f = open(filename)
    lines = f.readlines()
    f.close()
    lines = [line.rstrip() for line in lines]
    return lines

#------------------------------------------------------------------------------
def fulllist(path):
    '''
    import dclimate.d_std_lib as d_std_lib
    d_std_lib.fulllist(r'c:\python27')
    目录遍历
    '''
    for root, dirs, files in os.walk( path ):
        for fn in files:
            #print(root, fn)
            print(os.path.join(root, fn))


#简单小程序
################################################
#获取所在年月的日期
###############################################
def getMonths():

    from datetime import datetime
    import calendar

    d = datetime.now()
    c = calendar.Calendar()

    year = d.year
    month = d.month

    if month == 1 :
        month = 12
        year -= 1
    else :
        month -= 1
    months = calendar.monthrange(year, month)[1]
    return months

############################################################
#将GB2312变为utf-8
############################################################
def replaceXmlEncoding(filepath,f2, oldEncoding='GB2312', newEncoding='utf-8'):
    f = open(filepath, mode='r')
    content = f.read()
    content = re.sub(oldEncoding, newEncoding, content)
    #print(content.decode('gb2312'))
    b = content.decode('gb2312')
    b = b.encode('utf-8')
    #sys.exit(0)
    f.close()

    f = open(f2, mode='w')
    f.write(b)
    f.close()


############################################################
#将将关键字段返回数组
############################################################
def get_RegionID_by_XML(XmlFileName,RegionID_IN):
    from xml.dom import minidom
    xmldoc = minidom.parse(XmlFileName)
    #print(xmldoc.toxml())
    #a1 = xmldoc.getElementsByTagName('Object')[0]
    #print(a1)

    AllObjects  =  xmldoc.getElementsByTagName("Object")
    for  Object  in  AllObjects:
        ### print ( " ------------------------------------------- " )
        #exit(0)
        #nameNode  =  Object.getElementsByTagName("RegionID")[0]
        #print(nameNode.childNodes[0].nodeValue)
        RegionID = Object.getElementsByTagName("RegionID")[0].childNodes[0].nodeValue
        ### print(RegionID.encode('gb2312'))
        if(RegionID_IN!=RegionID):
            continue

        dict1={}
        RegionName = Object.getElementsByTagName("RegionName")[0].childNodes[0].nodeValue
        print(RegionID.encode('gb2312'))
        #比较笨，是用MINIDOM开发的，后期用 import xml.etree.ElementTree as etree 来修改
        #
        RegionShapeFile= Object.getElementsByTagName("RegionShapeFile")[0].childNodes[0].nodeValue
        ProjType= Object.getElementsByTagName("ProjType")[0].childNodes[0].nodeValue
        DrawArea= Object.getElementsByTagName("DrawArea")[0].childNodes[0].nodeValue
        RegionArea= Object.getElementsByTagName("RegionArea")[0].childNodes[0].nodeValue
        I_STA_TYPE= Object.getElementsByTagName("I_STA_TYPE")[0].childNodes[0].nodeValue

        LongitudeInfo= Object.getElementsByTagName("LongitudeInfo")[0].childNodes[0].nodeValue
        LatitudeInfo= Object.getElementsByTagName("LatitudeInfo")[0].childNodes[0].nodeValue
        StationInfoFile= Object.getElementsByTagName("StationInfoFile")[0].childNodes[0].nodeValue
        InterpToFile = Object.getElementsByTagName("InterpToFile")[0].childNodes[0].nodeValue

        Desc= Object.getElementsByTagName("Desc")[0].childNodes[0].nodeValue
        dict1['RegionID']=RegionID
        dict1['RegionName']=RegionName
        dict1['RegionShapeFile']=RegionShapeFile
        dict1['ProjType']=ProjType
        dict1['DrawArea']=DrawArea
        dict1['RegionArea']=RegionArea
        dict1['I_STA_TYPE']=I_STA_TYPE

        dict1['LongitudeInfo']=LongitudeInfo
        dict1['LatitudeInfo']=LatitudeInfo
        dict1['StationInfoFile']=StationInfoFile

        dict1['InterpToFile']=InterpToFile

        dict1['Desc']=Desc

        #print( Object.getElementsByTagName("ShapeFiles") == list() )
        #sys.exit(0)

        if([] == Object.getElementsByTagName("ShapeFiles")):
            #pass
            continue
        ShapeFiles  =  Object.getElementsByTagName("ShapeFiles")[0]
        FFF = ShapeFiles.getElementsByTagName("F")
        list1=[]
        for F1 in FFF:
            dict2={}
            Shapefile = F1.getElementsByTagName("ShapeFile")[0].childNodes[0].nodeValue
            COLOR = F1.getElementsByTagName("COLOR")[0].childNodes[0].nodeValue
            LineWidth = F1.getElementsByTagName("LineWidth")[0].childNodes[0].nodeValue
            dict2['Shapefile']=Shapefile
            dict2['COLOR']=COLOR
            dict2['LineWidth']=LineWidth
            #print(Shapefile,COLOR,LineWidth)
            list1.append(dict2)
            dict1['Shapefiles']=list1

        #print(dict1)
        return dict1


def GetStaInfo(FileName):
    X1,X2,Y1,Y2,ST,STID = [],[],[],[],[],[]
    f=open(FileName)
    List1 = f.readlines()
    for txt in List1:
        #print(txt)
        txt.strip()
        l1 = txt.split()
        STID.append(float(l1[0]))
        X1.append(float(l1[1]))
        Y1.append(float(l1[2]))
        ST.append(l1[3])
        X2.append(float(l1[4]))
        Y2.append(float(l1[5]))

    f.close
    return(np.array(X1),np.array(Y1),ST ,X2,Y2 )
    ###################################################



#起始经纬度范围
def str_to_x1y1x2y2(limitStr):
    list1 = limitStr.split(',');
    if(4==len(list1)):
        #print(list1)
        mid_lon = float(list1[0])+float(list1[1])/2.0
        return(float(list1[0]),float(list1[1]),float(list1[2]),float(list1[3]),None)
    #开始结束经纬度间隔
    if(5==len(list1)):
        return(float(list1[0]),float(list1[1]),float(list1[2]),float(list1[3]),float(list1[4]))

def str_to_x1x2x3(limitStr):
    list1 = limitStr.split(',');
    #print(list1)
    return(float(list1[0]),float(list1[1]),float(list1[2]))
    #获取站点信息


####################################################################
def DrawMapMain_XML_CFG(dict1,Region,Levfile='',Title='',imgfile='out.png', \
                        bShowImage=True,bDrawNumber=True,bDrawPoint=True):
    '''
    Setup: Generate data...
    '''
    #LongitudeInfo
    print('aaaaaaaaaa',dict1['RegionArea'])
    xmin,xmax,ymin,ymax,LonCenter = str_to_x1y1x2y2(dict1['RegionArea'])
    xmin2,xmax2,ymin2,ymax2,LonCenter = str_to_x1y1x2y2(dict1['DrawArea'])

    INITDIR = dict1['INITDIR']
    shpfile1=dict1['RegionShapeFile']
    shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)
    #print(shpfile1)

    print(LonCenter)
    if(None==LonCenter):
        LonCenter = (xmin+xmax)/2.0
    print(LonCenter)
    #sys.exit(0)

    shpfile_border=shpfile1
    print('Border shpfile=',shpfile1)
    #sys.exit(0)
    nx, ny = 150,150
    import os
    '''if(not os.path.isfile(inputfile)):
        print('输入文件%s不存在，请检查！'%(inputfile))
        print('import %s is not exist ,Please Check program EXIT!!！'%(inputfile))
        sys.exit(0)
    '''
    import numpy as np
    #Region = np.genfromtxt(inputfile)
    #print(Region)
    x,y,z = Region[:,0],Region[:,1],Region[:,2]
    #print(x,y,z)
    #sys.exit(0)

    #xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax =108,116.5,29,34  //90,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)


    #离散点插值到网格
    mytic()
    import dgriddata as dgrid
    if np.size(x)>250 :
        zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='Invdisttree')#Invdisttree
        #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='kriging')#Invdisttree
    else:
        if np.size(x)>30 :
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf')#scipy_idw')# #line_rbf  line_rbf2
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf  line_rbf2
        else:
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf2')#scipy_idw')# #line_rbf  line_rbf2 kriging
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf  line_rbf2
        #zi,xi,yi = df.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf

    mytoc('离散点插值到网格')

    #重要
    #http://hyry.dip.jp:8000/scipybook/default/file/03-scipy/scipy_interp2d.py
    mytic()

    #http://efreedom.com/Question/1-3526514/Problem-2D-Interpolation-SciPy-Non-Rectangular-Grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid

    mytic()
    zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=int(4)) #
    mytoc('扩展矩阵插值: ')
    #sys.exit(0)

    #获取mask矩阵
    mytic()
    #grid1,shapes = df.build_inside_mask_array(r"spatialdat\china_province",x1,y1)
    #grid1,shapes = df.build_inside_mask_array(r"spatialdata\hbsj-mian",x1,y1)
    #a1=print(a)

    grid1 = dgrid.build_inside_mask_array(shpfile1,x1,y1)
    mytoc('mask非绘图区域')
    #print('AAAA')
    zi[np.logical_not(grid1)]=np.NaN
    #提供空白底图使用
    #zi=np.NaN
    #zi[grid1]=np.NaN


    ############################################################################
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap,shapefile
    ############################################################################

    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    print(xmin2,ymin2,xmax2,ymax2)
    #sys.exit(0)
    ProjType=dict1['ProjType']

    if('lcc'==ProjType):
        m = Basemap(llcrnrlon=xmin2,llcrnrlat=ymin2,urcrnrlon=xmax2,urcrnrlat=ymax2, \
                    projection='lcc',lon_0=LonCenter,lat_0=30.0,lat_1=60.0)
    if('merc'==ProjType):
        #90,122.5,24.0,36.0
        m = Basemap(projection='merc',llcrnrlat=ymin2,urcrnrlat=ymax2, \
                    llcrnrlon=xmin2,urcrnrlon=xmax2,resolution='c')

    #m = Basemap(projection='merc',llcrnrlat=24,urcrnrlat=36,\
    #    llcrnrlon=90,urcrnrlon=122.5,resolution='c')
    m.ax=ax1

    lat1,lat2,lat3 = str_to_x1x2x3(dict1['LatitudeInfo'])
    m.drawparallels(np.arange(lat1,lat2,lat3),labels=[1,0,0,0],linewidth=0.3, fontsize=10)

    lon1,lon2,lon3 = str_to_x1x2x3(dict1['LongitudeInfo'])
    m.drawmeridians(np.arange(lon1,lon2,lon3),labels=[0,0,0,1],linewidth=0.3, fontsize=9)

    #%#m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    #%#m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    #m.drawcoastlines(linewidth=0.2)  #画海岸线
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()


    #mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    #m.readshapefile('CJ_GIS_INFO\CJ_Basin2_Line','',drawbounds=True,color='r',linewidth=0.2)
    #mytoc(u'画流域界: ')

    #mytic()
    #df.draw_map_lines(m,r"spatialdata\CJ_LVL1_RIVER",color='b',linewidth=1.0)
    #mytoc(u'画长江干流: ')

    list1 = dict1['Shapefiles']
    ### print('BBBBB')
    ### print(list1)
    ### print('AAAAAA')
    #color='#888888'

    for line1 in list1:
        ### print(line1['Shapefile'].encode('gb2312'),line1['COLOR'].encode('gb2312'),line1['LineWidth'].encode('gb2312'))
        shpfile1 = line1['Shapefile']
        #替换INITDIR至安装目录
        shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)

        shpfile0 = os.path.splitext(shpfile1)[0]
        COLOR = '#'+line1['COLOR']
        LineWidth1 = float(line1['LineWidth'])

        if(os.path.isfile(shpfile1)):
            m.readshapefile(shpfile0,'',drawbounds=True,color=COLOR,linewidth=LineWidth1)
            #dgrid.draw_map_lines(m,shpfile0,color=COLOR,linewidth=LineWidth1)
        #sys.exit(0)

    mytic()
    #df.draw_map_lines(m,shpfile1,color='k',linewidth=1.0)
    #df.draw_map_lines(m,shpfile1,color='g',linewidth=0.50)
    #print('shapefile=',shpfile1)
    #print(os.path.splitext(shpfile1)[0])
    #print(os.join.)

    #sys.exit(0)

    if(os.path.isfile(shpfile1)):
        m.readshapefile(os.path.splitext(shpfile1)[0],'',drawbounds=True,color='k',linewidth=0.2)
    mytoc('画省界')

    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
    #maplev = np.loadtxt(LevelFile)#'LEV\maplev_TA.LEV')

    #cmap2 = maplev[:,:-1]
    #cmap2 = cmap2/255.0
    #lev = maplev[:,-1]
    #print(cmap2)

    #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
    mytic()
    from __init__ import Spatial_Data,Station_Data,Level_Path

    if(not os.path.isfile(Levfile)):
        Levfile = os.path.join(Level_Path,Levfile)

    if( os.path.isfile(Levfile)):
        maplev = np.loadtxt(Levfile)
        #maplev = np.loadtxt('LEV\maplev_TA.LEV')

        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)

        cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        #l, b, w, h = pos.bounds
        #cax = plt.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
        #plt.colorbar(drawedges=True, cax=cax)
        plt.colorbar(cax,shrink=0.4)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
        #plt.colorbar(ax=ax1)
        CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        plt.clabel(CS, fmt='%4.1f',inline=0.5, fontsize=8)  #inline 为等值线标注值周围的间隔
    else:
        cax = m.contourf(xi,yi,zi)
        #CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        CS = m.contour(xi,yi,zi,linewidths=.6,colors='k')  #,levels=lev
        plt.colorbar(cax,shrink=0.4)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
        plt.clabel(CS, fmt='%4.1f',inline=0.75, fontsize=8)  #inline 为等值线标注值周围的间隔

    mytoc('fill contour')


    #############################################################
    #绘制站点信息
    import dclimate as dclim
    cnfont = dclim.GetCnFont2(11)

    StationInfoFile = dict1['StationInfoFile']

    StationInfoFile = StationInfoFile.replace('${INITDIR}',INITDIR)
    if(os.path.isfile(StationInfoFile)):
        xs,ys,staname,xs2,ys2 = GetStaInfo(StationInfoFile)
        xsm,ysm = m(xs,ys)
        xsm2,ysm2 = m(xs+xs2,ys+ys2)
        #m.plot(xsm,ysm,'o',color='00000',markersize=3,markeredgewidth=0.01)
        m.plot(xsm,ysm,'o',markersize=3,markeredgewidth=0.01)
        for ii in range(len(staname)):
            str1 = unicode(staname[ii],'gbk')
            plt.text(xsm2[ii],ysm2[ii],str1,fontsize=7,fontproperties=cnfont)
    #绘制站点信息
    ###############################################################

    xt1,yt1 = m(x-0.4,y-0.2)  #变换为proj.4
    x,y = m(x,y)  #变换为proj.4

    #m.plot(x,y,c=z,markeredgewidth=0.1)
    if(bDrawNumber):
    #画插值
        #x2,y = m(xnew,ynew)
        #画站点
        if(bDrawPoint):
            #m.scatter(x,y,s=4,lw=0)
            #cnfont = dclim.GetCnFont()
            #plt.plot(x,y,'o')
            #x2,y2 = m(x-0.3,x+0.04)
            m.scatter(x,y,c=z,s=4,lw=0)

        for i in range(len(x)):
            #zc2 为 插值后的实况值
            #print(x[i],y[i],z[i])
            #plt.text(x2[i],y2[i],'%5.1f'%zc2[i],fontsize=9)
            plt.text(x[i],y[i],'%5.1f'%z[i],fontsize=7)
            #plt.text(xt1[i],yt1[i],'%5.1f'%z[i]+' ',fontsize=9)
            #plt.text(x2[i],y2[i],'%5.0f'%zc3[i]+' ',fontsize=9,fontproperties=cnfont)
            #画站名

            #x2,y2 = m(xnew-0.1,ynew-0.1)
            #for i in range(len(x)):
            #    str1 = unicode(staname[i],'gbk')
            #    plt.text(x2[i],y2[i],str1,fontsize=6,fontproperties=cnfont)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    import dclimate as dclim
    cnfont = dclim.GetCnFont2(12)
    plt.title(Title.decode('gb2312'),fontproperties=cnfont)


    print('shpfile_border=',shpfile_border)

    #sys.exit(0)
    b_show_ss=False

    import re
    p = re.compile('china')
    match = p.search(shpfile_border)
    if(match):
        #b_show_ss=False
        b_show_ss=True

    #b_show_ss=False
    if(b_show_ss):

        mytic()
        ax2 = fig.add_axes([0.5866,0.149,0.2,0.2])
        m2 = Basemap(projection='cyl',llcrnrlat=4,urcrnrlat=25, \
                     llcrnrlon=107,urcrnrlon=122,resolution='h',ax=ax2)
        #sf = shapefile.Reader(r"spatialdat\Export_Output_12")
        mytoc('ee')
        southsea_file = os.path.join(Spatial_Data,"Export_Output_12")

        print(southsea_file)
        #import dfunc as df
        #df.draw_map_lines(m2,southsea_file,linewidth=0.5)
        mytic()
        m2.readshapefile(southsea_file,'',drawbounds=True,color='k',linewidth=0.5)
        #m.readshapefile(southsea_file,drawbounds=True,color='k',linewidth=0.5)
        mytoc('Draw South Sea')

    #plt.show()
    #plt.colorbar()
    #plt.savefig('03.png',dpi=150)
    print(imgfile)
    #sys.exit(0)
    #imgfile='a.png'
    plt.savefig(imgfile,dpi=150)
    import dfunc as df
    df.CutPicWhiteBorder(imgfile)
    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    if(bShowImage):
        plt.show()
    plt.close(fig)
    print('绘图结束!!!!!!!!!!')
    mytoc('绘图所耗时间')


####################################################################
#
####################################################################
def DrawMapMain_XML_CFG_Split_Cluster(dict1,Region,Levfile='',Title='',imgfile='out.png', \
                        bShowImage=True,bDrawNumber=False,cluster_num=0,b_show_ss=False):
    '''
    Setup: Generate data...
    '''
    #LongitudeInfo

    xmin,xmax,ymin,ymax,LonCenter = str_to_x1y1x2y2(dict1['RegionArea'])
    xmin2,xmax2,ymin2,ymax2,LonCenter = str_to_x1y1x2y2(dict1['DrawArea'])



    INITDIR = dict1['INITDIR']
    shpfile1=dict1['RegionShapeFile']
    shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)


    #sys.exit(0)
    print('LonCenter=',LonCenter)
    if(None==LonCenter):
        LonCenter = (xmin+xmax)/2.0
    print('LonCenter=',LonCenter)

    shpfile_border=shpfile1
    print('Border shpfile=',shpfile1)
    #sys.exit(0)
    if(cluster_num>0):
        nx, ny = 150*4,150*4
    else:
        nx, ny = 150,150
    #nx,ny=100,100

    import os
    '''if(not os.path.isfile(inputfile)):
        print('输入文件%s不存在，请检查！'%(inputfile))
        print('import %s is not exist ,Please Check program EXIT!!！'%(inputfile))
        sys.exit(0)
    '''
    import numpy as np
    #Region = np.genfromtxt(inputfile)
    #print(Region)
    x,y,z = Region[:,0],Region[:,1],Region[:,2]
    #print(x,y,z)
    #sys.exit(0)

    #xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax =108,116.5,29,34  //90,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)


    #离散点插值到网格
    mytic()
    import dgriddata as dgrid
    if np.size(x)>350 :
        zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#Invdisttree nearest
    else:
        if np.size(x)>40 :
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#scipy_idw')# #line_rbf  line_rbf2 line_rbf
        else:
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#scipy_idw')# #line_rbf  line_rbf2 kriging Invdisttree
            #zi,xi,yi = df.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf

    mytoc('离散点插值到网格')
    if cluster_num>0:
        zi.round()
    #重要
    #http://hyry.dip.jp:8000/scipybook/default/file/03-scipy/scipy_interp2d.py
    mytic()

    #http://efreedom.com/Question/1-3526514/Problem-2D-Interpolation-SciPy-Non-Rectangular-Grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid

    mytic()
    if (0==cluster_num):
        zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=int(2)) #
        zi.round()
    mytoc('扩展矩阵插值: ')
    #sys.exit(0)

    #获取mask矩阵
    mytic()
    #grid1,shapes = df.build_inside_mask_array(r"spatialdat\china_province",x1,y1)
    #grid1,shapes = df.build_inside_mask_array(r"spatialdata\hbsj-mian",x1,y1)
    #a1=print(a)

    grid1 = dgrid.build_inside_mask_array(shpfile1,x1,y1)
    mytoc('mask非绘图区域')
    zi[np.logical_not(grid1)]=np.NaN

    ############################################################################
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap,shapefile
    ############################################################################

    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    print(xmin2,ymin2,xmax2,ymax2)
    #sys.exit(0)
    ProjType=dict1['ProjType']

    if('lcc'==ProjType):
        m = Basemap(llcrnrlon=xmin2,llcrnrlat=ymin2,urcrnrlon=xmax2,urcrnrlat=ymax2, \
                    projection='lcc',lon_0=LonCenter,lat_0=30.0,lat_1=60.0,resolution='l')
    if('merc'==ProjType):
        #90,122.5,24.0,36.0
        m = Basemap(projection='merc',llcrnrlat=ymin2,urcrnrlat=ymax2, \
                    llcrnrlon=xmin2,urcrnrlon=xmax2,resolution='c')

    #m = Basemap(projection='merc',llcrnrlat=24,urcrnrlat=36,\
    #    llcrnrlon=90,urcrnrlon=122.5,resolution='c')
    m.ax=ax1

    lat1,lat2,lat3 = str_to_x1x2x3(dict1['LatitudeInfo'])
    m.drawparallels(np.arange(lat1,lat2,lat3),labels=[1,0,0,0],linewidth=0.3, fontsize=10)

    lon1,lon2,lon3 = str_to_x1x2x3(dict1['LongitudeInfo'])
    m.drawmeridians(np.arange(lon1,lon2,lon3),labels=[0,0,0,1],linewidth=0.3, fontsize=9)

    #%#m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    #%#m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    m.drawcoastlines(linewidth=0.4)  #画海岸线 此处可以不绘制

    #m.drawcountries(linewidth=0.4)
    m.drawmapboundary()


    #mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    #m.readshapefile('CJ_GIS_INFO\CJ_Basin2_Line','',drawbounds=True,color='r',linewidth=0.2)
    #mytoc(u'画流域界: ')

    #mytic()
    #df.draw_map_lines(m,r"spatialdata\CJ_LVL1_RIVER",color='b',linewidth=1.0)
    #mytoc(u'画长江干流: ')

    list1 = dict1['Shapefiles']
    #print(list1)
    #color='#888888'

    for line1 in list1:
        #print(line1['Shapefile'],line1['COLOR'],line1['LineWidth'])
        shpfile1 = line1['Shapefile']
        shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)



        shpfile0 = os.path.splitext(shpfile1)[0]
        COLOR = '#'+line1['COLOR']
        LineWidth1 = float(line1['LineWidth'])

        if(os.path.isfile(shpfile1)):
            m.readshapefile(shpfile0,'',drawbounds=True,color=COLOR,linewidth=LineWidth1)
            #dgrid.draw_map_lines(m,shpfile0,color=COLOR,linewidth=LineWidth1)
            #sys.exit(0)

    mytic()
    #df.draw_map_lines(m,shpfile1,color='k',linewidth=1.0)
    #df.draw_map_lines(m,shpfile1,color='g',linewidth=0.50)
    #print('shapefile=',shpfile1)
    #print(os.path.splitext(shpfile1)[0])
    #print(os.join.)

    #sys.exit(0)

    if(os.path.isfile(shpfile1)):
        m.readshapefile(os.path.splitext(shpfile1)[0],'',drawbounds=True,color='k',linewidth=0.2)
    mytoc('画省界')

    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
    #maplev = np.loadtxt(LevelFile)#'LEV\maplev_TA.LEV')

    #cmap2 = maplev[:,:-1]
    #cmap2 = cmap2/255.0
    #lev = maplev[:,-1]
    #print(cmap2)

    #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
    from __init__ import Spatial_Data,Station_Data,Level_Path

    if(not os.path.isfile(Levfile)):
        Levfile = os.path.join(Level_Path,Levfile)

    if( os.path.isfile(Levfile)):
        maplev = np.loadtxt(Levfile)
        #maplev = np.loadtxt('LEV\maplev_TA.LEV')

        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)

        cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        #l, b, w, h = pos.bounds
        #cax = plt.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
        #plt.colorbar(drawedges=True, cax=cax)
        plt.colorbar(cax,shrink=0.3)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
        #plt.colorbar(ax=ax1)
        CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        plt.clabel(CS, fmt='%4.1f',inline=0.5, fontsize=8)  #inline 为等值线标注值周围的间隔
    else:
        #pass
        #'''
        if cluster_num>0:
            cmap = plt.get_cmap('gist_rainbow',cluster_num) #Pastel1
            #cmap.set_bad(color='w',alpha=1)
            zi = np.ma.masked_invalid(zi)
            iml =m.pcolormesh(xi,yi,zi,cmap=cmap, norm=plt.Normalize(1,cluster_num+1))
            cb = m.colorbar(iml,"bottom", size="2%", pad="1%")
            cb.set_ticks(np.arange(1,cluster_num+1))
        else:

            cax = m.contourf(xi,yi,zi)
            #CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
            CS = m.contour(xi,yi,zi,linewidths=.6,colors='k')  #,levels=lev
            plt.colorbar(cax,shrink=0.3)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
            plt.clabel(CS, fmt='%4.1f',inline=0.75, fontsize=8)  #inline 为等值线标注值周围的间隔


    #############################################################
    #绘制站点信息
    import dclimate as dclim
    cnfont = dclim.GetCnFont()

    StationInfoFile = dict1['StationInfoFile']
    if(os.path.isfile(StationInfoFile)):
        xs,ys,staname,xs2,ys2 = GetStaInfo(StationInfoFile)
        xsm,ysm = m(xs,ys)
        xsm2,ysm2 = m(xs+xs2,ys+ys2)
        #m.plot(xsm,ysm,'o',color='00000',markersize=3,markeredgewidth=0.01)
        m.plot(xsm,ysm,'o',markersize=1,markeredgewidth=0.01)
        for ii in range(len(staname)):
            str1 = unicode(staname[ii],'gbk')
            plt.text(xsm2[ii],ysm2[ii],str1,fontsize=6,fontproperties=cnfont)
        #绘制站点信息
    ###############################################################

    xt1,yt1 = m(x-0.4,y-0.2)  #变换为proj.4
    x,y = m(x,y)  #变换为proj.4

    #m.plot(x,y,c=z,markeredgewidth=0.1)
    if(bDrawNumber):
    #画插值
        #x2,y = m(xnew,ynew)
        #画站点
        m.scatter(x,y,s=4,lw=0)
        #plt.plot(x,y,'o')
        #x2,y2 = m(x-0.3,x+0.04)
        #--  m.scatter(x,y,c=z,s=4,lw=0)
        cnfont = dclim.GetCnFont2(size1=8)
        for i in range(len(x)):
            #zc2 为 插值后的实况值
            #print(x[i],y[i],z[i])
            #plt.text(x2[i],y2[i],'%5.1f'%zc2[i],fontsize=9)
            plt.text(xt1[i],yt1[i],'%d'%z[i],fontsize=8)
            #plt.text(xt1[i],yt1[i],'%5.1f'%z[i]+' ',fontsize=9)
            #plt.text(x2[i],y2[i],'%5.0f'%zc3[i]+' ',fontsize=9,fontproperties=cnfont)
            #画站名

            #x2,y2 = m(xnew-0.1,ynew-0.1)
            #for i in range(len(x)):
            #    str1 = unicode(staname[i],'gbk')
            #    plt.text(x2[i],y2[i],str1,fontsize=6,fontproperties=cnfont)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    import dclimate as dclim
    cnfont = dclim.GetCnFont2(12)
    plt.title(Title.decode('gb2312'),fontproperties=cnfont)
    #plt.title(Title,size=12)


    print('shpfile_border=',shpfile_border)

    #sys.exit(0)


    # import re
    # p = re.compile('china')
    # match = p.search(shpfile_border)
    # if(match):
    #     #b_show_ss=False
    #     b_show_ss=True


    if(b_show_ss):

        mytic()
        ax2 = fig.add_axes([0.7,0.124,0.2,0.2])
        m2 = Basemap(projection='cyl',llcrnrlat=4,urcrnrlat=25, \
                     llcrnrlon=107,urcrnrlon=122,resolution='h',ax=ax2)
        #sf = shapefile.Reader(r"spatialdat\Export_Output_12")
        mytoc('ee')
        southsea_file = os.path.join(Spatial_Data,"Export_Output_12")

        print(southsea_file)
        #import dfunc as df
        #df.draw_map_lines(m2,southsea_file,linewidth=0.5)
        mytic()
        m2.readshapefile(southsea_file,'',drawbounds=True,color='k',linewidth=0.5)
        #m.readshapefile(southsea_file,drawbounds=True,color='k',linewidth=0.5)
        mytoc('Draw South Sea')

    #plt.colorbar()
    #plt.savefig('03.png',dpi=150)
    plt.savefig(imgfile, dpi=180)

    #plt.savefig(imgfile.replace('.png','.svg'))
    # plt.savefig(imgfile.replace('.png','.pdf'))
    # plt.savefig(imgfile.replace('.png','.ps'))
    # plt.savefig(imgfile.replace('.png','.eps'))

    import dfunc as df
    df.CutPicWhiteBorder(imgfile)
    ##plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    if(bShowImage):
        plt.show()
    plt.close(fig)
    print('绘图结束!!!!!!!!!!')
    mytoc('绘图所耗时间')




def DrawMapMain_XML_CFG_Split_Cluster_OneArea(dict1,Region,Levfile='',Title='',imgfile='out.png', \
                                      bShowImage=True,bDrawNumber=False,cluster_num=0,LabelCode=1,b_show_ss=False):
    '''
    只绘制一个区域，其他的不绘制
    Setup: Generate data...
    '''
    #LongitudeInfo

    xmin,xmax,ymin,ymax,LonCenter = str_to_x1y1x2y2(dict1['RegionArea'])
    xmin2,xmax2,ymin2,ymax2,LonCenter = str_to_x1y1x2y2(dict1['DrawArea'])



    INITDIR = dict1['INITDIR']
    shpfile1=dict1['RegionShapeFile']
    shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)


    #sys.exit(0)
    if(None==LonCenter):
        LonCenter = (xmin+xmax)/2.0
        #LonCenter = (xmin+xmax)/2.0

    shpfile_border=shpfile1
    print('Border shpfile=',shpfile1)
    #sys.exit(0)
    if(cluster_num>0):
        nx, ny = 250*3,250*3
    else:
        nx, ny = 250,250
    import os
    '''if(not os.path.isfile(inputfile)):
        print('输入文件%s不存在，请检查！'%(inputfile))
        print('import %s is not exist ,Please Check program EXIT!!！'%(inputfile))
        sys.exit(0)
    '''
    import numpy as np
    #Region = np.genfromtxt(inputfile)
    #print(Region)
    x,y,z = Region[:,0],Region[:,1],Region[:,2]
    #print(x,y,z)
    #sys.exit(0)

    #xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax =108,116.5,29,34  //90,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)


    #离散点插值到网格
    mytic()
    import dgriddata as dgrid
    if np.size(x)>350 :
        zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#Invdisttree nearest
    else:
        if np.size(x)>40 :
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#scipy_idw')# #line_rbf  line_rbf2 line_rbf
        else:
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#scipy_idw')# #line_rbf  line_rbf2 kriging Invdisttree
            #zi,xi,yi = df.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf

    mytoc('离散点插值到网格')
    if cluster_num>0:
        zi.round()
        #重要
    #http://hyry.dip.jp:8000/scipybook/default/file/03-scipy/scipy_interp2d.py
    mytic()

    #http://efreedom.com/Question/1-3526514/Problem-2D-Interpolation-SciPy-Non-Rectangular-Grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid

    mytic()
    if (0==cluster_num):
        zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=int(2)) #
        zi.round()
    mytoc('扩展矩阵插值: ')
    #sys.exit(0)

    #获取mask矩阵
    mytic()
    #grid1,shapes = df.build_inside_mask_array(r"spatialdat\china_province",x1,y1)
    #grid1,shapes = df.build_inside_mask_array(r"spatialdata\hbsj-mian",x1,y1)
    #a1=print(a)

    grid1 = dgrid.build_inside_mask_array(shpfile1,x1,y1)
    mytoc('mask非绘图区域')
    zi[np.logical_not(grid1)]=np.NaN
    zi = np.where(zi!=LabelCode,np.NaN,zi)

    #重要


    ############################################################################
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap,shapefile
    ############################################################################

    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    print(xmin2,ymin2,xmax2,ymax2)
    #sys.exit(0)
    ProjType=dict1['ProjType']

    if('lcc'==ProjType):
        m = Basemap(llcrnrlon=xmin2,llcrnrlat=ymin2,urcrnrlon=xmax2,urcrnrlat=ymax2, \
                    projection='lcc',lon_0=LonCenter,lat_0=30.0,lat_1=60.0)
    if('merc'==ProjType):
        #90,122.5,24.0,36.0
        m = Basemap(projection='merc',llcrnrlat=ymin2,urcrnrlat=ymax2, \
                    llcrnrlon=xmin2,urcrnrlon=xmax2,resolution='c')

    #m = Basemap(projection='merc',llcrnrlat=24,urcrnrlat=36,\
    #    llcrnrlon=90,urcrnrlon=122.5,resolution='c')
    m.ax=ax1

    lat1,lat2,lat3 = str_to_x1x2x3(dict1['LatitudeInfo'])
    m.drawparallels(np.arange(lat1,lat2,lat3),labels=[1,0,0,0],linewidth=0.3, fontsize=10)

    lon1,lon2,lon3 = str_to_x1x2x3(dict1['LongitudeInfo'])
    m.drawmeridians(np.arange(lon1,lon2,lon3),labels=[0,0,0,1],linewidth=0.3, fontsize=9)

    #%#m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    #%#m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    #m.drawcoastlines(linewidth=0.2)  #画海岸线
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()


    #mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    #m.readshapefile('CJ_GIS_INFO\CJ_Basin2_Line','',drawbounds=True,color='r',linewidth=0.2)
    #mytoc(u'画流域界: ')

    #mytic()
    #df.draw_map_lines(m,r"spatialdata\CJ_LVL1_RIVER",color='b',linewidth=1.0)
    #mytoc(u'画长江干流: ')

    list1 = dict1['Shapefiles']
    print(list1)
    #color='#888888'

    for line1 in list1:
        print(line1['Shapefile'],line1['COLOR'],line1['LineWidth'])
        shpfile1 = line1['Shapefile']
        shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)



        shpfile0 = os.path.splitext(shpfile1)[0]
        COLOR = '#'+line1['COLOR']
        LineWidth1 = float(line1['LineWidth'])

        if(os.path.isfile(shpfile1)):
            m.readshapefile(shpfile0,'',drawbounds=True,color=COLOR,linewidth=LineWidth1)
            #dgrid.draw_map_lines(m,shpfile0,color=COLOR,linewidth=LineWidth1)
            #sys.exit(0)

    mytic()
    #df.draw_map_lines(m,shpfile1,color='k',linewidth=1.0)
    #df.draw_map_lines(m,shpfile1,color='g',linewidth=0.50)
    #print('shapefile=',shpfile1)
    #print(os.path.splitext(shpfile1)[0])
    #print(os.join.)

    #sys.exit(0)

    if(os.path.isfile(shpfile1)):
        m.readshapefile(os.path.splitext(shpfile1)[0],'',drawbounds=True,color='k',linewidth=0.2)
    mytoc('画省界')

    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
    #maplev = np.loadtxt(LevelFile)#'LEV\maplev_TA.LEV')

    #cmap2 = maplev[:,:-1]
    #cmap2 = cmap2/255.0
    #lev = maplev[:,-1]
    #print(cmap2)

    #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
    from __init__ import Spatial_Data,Station_Data,Level_Path

    if(not os.path.isfile(Levfile)):
        Levfile = os.path.join(Level_Path,Levfile)

    if( os.path.isfile(Levfile)):
        maplev = np.loadtxt(Levfile)
        #maplev = np.loadtxt('LEV\maplev_TA.LEV')

        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)

        cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        #l, b, w, h = pos.bounds
        #cax = plt.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
        #plt.colorbar(drawedges=True, cax=cax)
        plt.colorbar(cax,shrink=0.3)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
        #plt.colorbar(ax=ax1)
        CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        plt.clabel(CS, fmt='%4.1f',inline=0.5, fontsize=8)  #inline 为等值线标注值周围的间隔
    else:
        #pass
        #'''
        if cluster_num>0:
            cmap = plt.get_cmap('gist_rainbow',cluster_num) #Pastel1
            #cmap.set_bad(color='w',alpha=1)
            zi = np.ma.masked_invalid(zi)

            #print(zi)
            #sys.exit(0)

            iml =m.pcolormesh(xi,yi,zi,cmap=cmap, norm=plt.Normalize(1,cluster_num+1))
            cb = m.colorbar(iml,"bottom", size="2%", pad="1%")
            cb.set_ticks(np.arange(1,cluster_num+1))
        else:

            cax = m.contourf(xi,yi,zi)
            #CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
            CS = m.contour(xi,yi,zi,linewidths=.6,colors='k')  #,levels=lev
            plt.colorbar(cax,shrink=0.3)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
            plt.clabel(CS, fmt='%4.1f',inline=0.75, fontsize=8)  #inline 为等值线标注值周围的间隔






    #############################################################
    #绘制站点信息
    import dclimate as dclim
    cnfont = dclim.GetCnFont()

    StationInfoFile = dict1['StationInfoFile']
    if(os.path.isfile(StationInfoFile)):
        xs,ys,staname,xs2,ys2 = GetStaInfo(StationInfoFile)
        xsm,ysm = m(xs,ys)
        xsm2,ysm2 = m(xs+xs2,ys+ys2)
        #m.plot(xsm,ysm,'o',color='00000',markersize=3,markeredgewidth=0.01)
        m.plot(xsm,ysm,'o',markersize=1,markeredgewidth=0.01)
        for ii in range(len(staname)):
            str1 = unicode(staname[ii],'gbk')
            plt.text(xsm2[ii],ysm2[ii],str1,fontsize=6,fontproperties=cnfont)
            #绘制站点信息
        ###############################################################

    xt1,yt1 = m(x-0.4,y-0.2)  #变换为proj.4
    x,y = m(x,y)  #变换为proj.4

    #m.plot(x,y,c=z,markeredgewidth=0.1)
    if(bDrawNumber):
    #画插值
        #x2,y = m(xnew,ynew)
        #画站点
        m.scatter(x,y,s=4,lw=0)
        #plt.plot(x,y,'o')
        #x2,y2 = m(x-0.3,x+0.04)
        #--  m.scatter(x,y,c=z,s=4,lw=0)
        cnfont = dclim.GetCnFont2(size1=8)
        for i in range(len(x)):
            #zc2 为 插值后的实况值
            #print(x[i],y[i],z[i])
            #plt.text(x2[i],y2[i],'%5.1f'%zc2[i],fontsize=9)
            plt.text(xt1[i],yt1[i],'%d'%z[i],fontsize=8)
            #plt.text(xt1[i],yt1[i],'%5.1f'%z[i]+' ',fontsize=9)
            #plt.text(x2[i],y2[i],'%5.0f'%zc3[i]+' ',fontsize=9,fontproperties=cnfont)
            #画站名

            #x2,y2 = m(xnew-0.1,ynew-0.1)
            #for i in range(len(x)):
            #    str1 = unicode(staname[i],'gbk')
            #    plt.text(x2[i],y2[i],str1,fontsize=6,fontproperties=cnfont)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    import dclimate as dclim
    cnfont = dclim.GetCnFont2(12)
    plt.title(Title.decode('gb2312'),fontproperties=cnfont)


    print('shpfile_border=',shpfile_border)

    #sys.exit(0)


    # import re
    # p = re.compile('china')
    # match = p.search(shpfile_border)
    # if(match):
    #     #b_show_ss=False
    #     #     b_show_ss=True
    b_show_ss=False


    if(b_show_ss):

        mytic()
        ax2 = fig.add_axes([0.7,0.130,0.2,0.2])
        m2 = Basemap(projection='cyl',llcrnrlat=4,urcrnrlat=25, \
                     llcrnrlon=107,urcrnrlon=122,resolution='h',ax=ax2)
        #sf = shapefile.Reader(r"spatialdat\Export_Output_12")
        mytoc('ee')
        southsea_file = os.path.join(Spatial_Data,"Export_Output_12")

        print(southsea_file)
        #import dfunc as df
        #df.draw_map_lines(m2,southsea_file,linewidth=0.5)
        mytic()
        m2.readshapefile(southsea_file,'',drawbounds=True,color='k',linewidth=0.5)
        #m.readshapefile(southsea_file,drawbounds=True,color='k',linewidth=0.5)
        mytoc('Draw South Sea')

    #plt.colorbar()
    #plt.savefig('03.png',dpi=150)
    plt.savefig(imgfile, dpi=180)
    import dfunc as df
    df.CutPicWhiteBorder(imgfile)
    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    if(bShowImage):
        plt.show()
    plt.close(fig)
    print('绘图结束!!!!!!!!!!')
    mytoc('绘图所耗时间')


########################################################
#生成预报日期字符串
########################################################
def Get_Pred_Date_String(Pred_Year,Pred_Mon,MonthCount):
    from datetime import datetime
    Pred_Year,Pred_Mon,MonthCount=int(Pred_Year),int(Pred_Mon),int(MonthCount)
    if(MonthCount==1):
        return '%d年%d月'%(Pred_Year,Pred_Mon)

    if(MonthCount>1):
        date1 = datetime.strptime('%04d-%02d-01'%(Pred_Year,Pred_Mon),'%Y-%m-%d')
        date2 = date1 + relativedelta(months=MonthCount-1)
        if (datetime.strftime(date2,'%Y')>datetime.strftime(date1,'%Y')):
            s1  =  '%d年%d月'%(Pred_Year,Pred_Mon)
            s2=datetime.strftime(date2,'%Y年')+'%d月'%date2.month
        else:
            #s2 = datetime.strftime(date2,'%m月')
            s1 = '%d年%d'%(Pred_Year,Pred_Mon)
            s2 = '%d月'%date2.month
        return s1+'-'+s2

########################################################
#生成预报日期字符串
########################################################
def draw_Region_Map(sky_region_config,RegionID,RegionDat,Levfile='maplev_RAP.lev',Title='',imgfile2='a.png',sShowInfo=[0,1,0]):#,Title='重庆夏季降水距平百分率例子'):

    #,bShowImage=bShowImage,bDrawNumber=bDrawNumber2
    bShowImage1 = sShowInfo[0]
    bDrawNumber1 = sShowInfo[1]
    bDrawPoint1 = sShowInfo[2]


    SKYCLIM1_HOME = os.environ.get('SKYCLIM1_HOME')

    #sky_region_config = os.path.join(SKYCLIM1_HOME,'config','sky_region_config.xml')
    tmp_config =  os.path.join(SKYCLIM1_HOME,'tmp','sky_reg_tmp.xml')

    replaceXmlEncoding(sky_region_config,tmp_config)
    #dict2 = d_std_lib.get_RegionID_by_XML('2.xml',u'湖北省76站ccccc')
    #dict2 = d_std_lib.get_RegionID_by_XML('2.xml',u'湖北省76站01')

    dict2 = get_RegionID_by_XML(tmp_config,RegionID.decode('gb2312'))
    sRegionID = dict2['RegionID']
    #print(sRegionID)
    #print(dict2)
    #file1 = os.path.join('tmp',  RegionID+'.png' )
    dict2['INITDIR']=SKYCLIM1_HOME
    #用于替换初始化目录

    #print('*'*79)
    #print(dict2)
    #sys.exit(0)
    DrawMapMain_XML_CFG(dict2,RegionDat,Levfile=Levfile,Title=Title,imgfile=imgfile2, \
                                  bShowImage=bShowImage1,bDrawNumber=bDrawNumber1,bDrawPoint=bDrawPoint1)#,Title='重庆夏季降水距平百分率例子')


def draw_image3(Region,Title1,LevFile1,OutPicFile1):
    SKYCLIM1_HOME = os.environ.get('SKYCLIM1_HOME')
    print(SKYCLIM1_HOME)
    '''
    import numpy as np
    a = np.random.random(10000)
    print(a)
    df.save_obj(a,'a.npy')
    b=df.load_obj('a.npy')
    print(b)
    sys.exit(0)
    '''
    sky_region_config = os.path.join(SKYCLIM1_HOME,'config','sky_region_config.xml')
    tmp_config =  os.path.join(SKYCLIM1_HOME,'tmp','sky_reg_tmp.xml')
    RegionID_File = os.path.join(SKYCLIM1_HOME,'config',"RegionID.ini")
    print('RegionID_File=',RegionID_File)
    replaceXmlEncoding(sky_region_config,tmp_config)
    #dict2 = d_std_lib.get_RegionID_by_XML('2.xml',u'湖北省76站ccccc')
    #dict2 = d_std_lib.get_RegionID_by_XML('2.xml',u'湖北省76站01')

    import dict4ini
    #config2 = ConfigObj(RegionID_File,encoding= "cp936")
    config2=dict4ini.DictIni(RegionID_File,commentdelimeter=';')

    #config2 = ConfigParser.ConfigParser()
    #config2.optionxform = str
    #config2.read(RegionID_File)

    RegionID = config2['GLOBAL']['RegionID']
    print('RegionID=',RegionID)
    #sys.exit()

    dict2 = get_RegionID_by_XML(tmp_config,RegionID.decode('gb2312'))
    print(dict2)
    print(dict2['RegionID'])
    print(dict2)
    if(not os.path.isdir('tmp')):
        os.mkdir("tmp")

    #OutPicFile1 = os.path.join(SKYCLIM1_HOME,'Output','Preferences',RegionID+'.png')


    dict2['INITDIR']=SKYCLIM1_HOME

    print('*'*79)
    print(dict2)
    #sys.exit(0)
    DrawMapMain_XML_CFG(dict2,Region,Levfile=LevFile1,Title=Title1,bDrawNumber=True,\
                                  imgfile=OutPicFile1,bShowImage=False)#,Title='重庆夏季降水距平百分率例子')



##########################################################################################
#
##########################################################################################
def DrawMapMain_XML_CFG_NoContour(dict1,Region,Levfile='',Title='',imgfile='out.png', \
                        bShowImage=True,bDrawNumber=True,bDrawPoint=True):
    '''
    Setup: Generate data...
    '''
    #LongitudeInfo
    print('aaaaaaaaaa',dict1['RegionArea'])
    xmin,xmax,ymin,ymax,LonCenter = str_to_x1y1x2y2(dict1['RegionArea'])
    xmin2,xmax2,ymin2,ymax2,LonCenter = str_to_x1y1x2y2(dict1['DrawArea'])

    INITDIR = dict1['INITDIR']
    shpfile1=dict1['RegionShapeFile']
    shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)
    #print(shpfile1)

    print(LonCenter)
    if(None==LonCenter):
        LonCenter = (xmin+xmax)/2.0
    print(LonCenter)
    #sys.exit(0)

    shpfile_border=shpfile1
    print('Border shpfile=',shpfile1)
    #sys.exit(0)
    nx, ny = 150,150
    import os
    '''if(not os.path.isfile(inputfile)):
        print('输入文件%s不存在，请检查！'%(inputfile))
        print('import %s is not exist ,Please Check program EXIT!!！'%(inputfile))
        sys.exit(0)
    '''
    import numpy as np
    #Region = np.genfromtxt(inputfile)
    #print(Region)
    x,y,z = Region[:,0],Region[:,1],Region[:,2]
    #print(x,y,z)
    #sys.exit(0)

    #xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax =108,116.5,29,34  //90,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)


    #离散点插值到网格
    mytic()
    import dgriddata as dgrid
    if np.size(x)>250 :
        zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='Invdisttree')#Invdisttree
        #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='kriging')#Invdisttree
    else:
        if np.size(x)>30 :
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf')#scipy_idw')# #line_rbf  line_rbf2
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf  line_rbf2
        else:
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf2')#scipy_idw')# #line_rbf  line_rbf2 kriging
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf  line_rbf2
        #zi,xi,yi = df.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf

    mytoc('离散点插值到网格')

    #重要
    #http://hyry.dip.jp:8000/scipybook/default/file/03-scipy/scipy_interp2d.py
    mytic()

    #http://efreedom.com/Question/1-3526514/Problem-2D-Interpolation-SciPy-Non-Rectangular-Grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid

    mytic()
    zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=int(4)) #
    mytoc('扩展矩阵插值: ')
    #sys.exit(0)

    #获取mask矩阵
    mytic()
    #grid1,shapes = df.build_inside_mask_array(r"spatialdat\china_province",x1,y1)
    #grid1,shapes = df.build_inside_mask_array(r"spatialdata\hbsj-mian",x1,y1)
    #a1=print(a)

    grid1 = dgrid.build_inside_mask_array(shpfile1,x1,y1)
    mytoc('mask非绘图区域')
    #print('AAAA')
    zi[np.logical_not(grid1)]=np.NaN
    #提供空白底图使用
    #zi=np.NaN
    #zi[grid1]=np.NaN


    ############################################################################
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap,shapefile
    ############################################################################

    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    print(xmin2,ymin2,xmax2,ymax2)
    #sys.exit(0)
    ProjType=dict1['ProjType']

    if('lcc'==ProjType):
        m = Basemap(llcrnrlon=xmin2,llcrnrlat=ymin2,urcrnrlon=xmax2,urcrnrlat=ymax2, \
                    projection='lcc',lon_0=LonCenter,lat_0=30.0,lat_1=60.0)
    if('merc'==ProjType):
        #90,122.5,24.0,36.0
        m = Basemap(projection='merc',llcrnrlat=ymin2,urcrnrlat=ymax2, \
                    llcrnrlon=xmin2,urcrnrlon=xmax2,resolution='c')

    #m = Basemap(projection='merc',llcrnrlat=24,urcrnrlat=36,\
    #    llcrnrlon=90,urcrnrlon=122.5,resolution='c')
    m.ax=ax1

    lat1,lat2,lat3 = str_to_x1x2x3(dict1['LatitudeInfo'])
    m.drawparallels(np.arange(lat1,lat2,lat3),labels=[1,0,0,0],linewidth=0.3, fontsize=10)

    lon1,lon2,lon3 = str_to_x1x2x3(dict1['LongitudeInfo'])
    m.drawmeridians(np.arange(lon1,lon2,lon3),labels=[0,0,0,1],linewidth=0.3, fontsize=9)

    #%#m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    #%#m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    #m.drawcoastlines(linewidth=0.2)  #画海岸线
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()


    #mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    #m.readshapefile('CJ_GIS_INFO\CJ_Basin2_Line','',drawbounds=True,color='r',linewidth=0.2)
    #mytoc(u'画流域界: ')

    #mytic()
    #df.draw_map_lines(m,r"spatialdata\CJ_LVL1_RIVER",color='b',linewidth=1.0)
    #mytoc(u'画长江干流: ')

    list1 = dict1['Shapefiles']
    ### print('BBBBB')
    ### print(list1)
    ### print('AAAAAA')
    #color='#888888'

    for line1 in list1:
        ### print(line1['Shapefile'].encode('gb2312'),line1['COLOR'].encode('gb2312'),line1['LineWidth'].encode('gb2312'))
        shpfile1 = line1['Shapefile']
        #替换INITDIR至安装目录
        shpfile1 = shpfile1.replace('${INITDIR}',INITDIR)

        shpfile0 = os.path.splitext(shpfile1)[0]
        COLOR = '#'+line1['COLOR']
        LineWidth1 = float(line1['LineWidth'])

        if(os.path.isfile(shpfile1)):
            m.readshapefile(shpfile0,'',drawbounds=True,color=COLOR,linewidth=LineWidth1)
            #dgrid.draw_map_lines(m,shpfile0,color=COLOR,linewidth=LineWidth1)
        #sys.exit(0)

    mytic()
    #df.draw_map_lines(m,shpfile1,color='k',linewidth=1.0)
    #df.draw_map_lines(m,shpfile1,color='g',linewidth=0.50)
    #print('shapefile=',shpfile1)
    #print(os.path.splitext(shpfile1)[0])
    #print(os.join.)

    #sys.exit(0)

    if(os.path.isfile(shpfile1)):
        m.readshapefile(os.path.splitext(shpfile1)[0],'',drawbounds=True,color='k',linewidth=0.2)
    mytoc('画省界')

    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
    #maplev = np.loadtxt(LevelFile)#'LEV\maplev_TA.LEV')

    #cmap2 = maplev[:,:-1]
    #cmap2 = cmap2/255.0
    #lev = maplev[:,-1]
    #print(cmap2)

    #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
    mytic()
    from __init__ import Spatial_Data,Station_Data,Level_Path

    if(not os.path.isfile(Levfile)):
        Levfile = os.path.join(Level_Path,Levfile)

    if( os.path.isfile(Levfile)):
        maplev = np.loadtxt(Levfile)
        #maplev = np.loadtxt('LEV\maplev_TA.LEV')

        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)

        ### cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        #l, b, w, h = pos.bounds
        #cax = plt.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
        #plt.colorbar(drawedges=True, cax=cax)

        ### plt.colorbar(cax,shrink=0.4)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
        #plt.colorbar(ax=ax1)

        ### CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        ### plt.clabel(CS, fmt='%4.1f',inline=0.5, fontsize=8)  #inline 为等值线标注值周围的间隔
    else:
        pass
        ### cax = m.contourf(xi,yi,zi)
        #CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        ### CS = m.contour(xi,yi,zi,linewidths=.6,colors='k')  #,levels=lev
        ### plt.colorbar(cax,shrink=0.4)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
        ### plt.clabel(CS, fmt='%4.1f',inline=0.75, fontsize=8)  #inline 为等值线标注值周围的间隔

    mytoc('fill contour')


    #############################################################
    #绘制站点信息
    import dclimate as dclim
    cnfont = dclim.GetCnFont2(11)

    StationInfoFile = dict1['StationInfoFile']

    StationInfoFile = StationInfoFile.replace('${INITDIR}',INITDIR)
    if(os.path.isfile(StationInfoFile)):
        xs,ys,staname,xs2,ys2 = GetStaInfo(StationInfoFile)
        xsm,ysm = m(xs,ys)
        xsm2,ysm2 = m(xs+xs2,ys+ys2)
        #m.plot(xsm,ysm,'o',color='00000',markersize=3,markeredgewidth=0.01)
        m.plot(xsm,ysm,'o',markersize=3,markeredgewidth=0.01)
        for ii in range(len(staname)):
            str1 = unicode(staname[ii],'gbk')
            plt.text(xsm2[ii],ysm2[ii],str1,fontsize=11,fontproperties=cnfont)
    #绘制站点信息
    ###############################################################

    xt1,yt1 = m(x-0.4,y-0.2)  #变换为proj.4
    x,y = m(x,y)  #变换为proj.4

    #m.plot(x,y,c=z,markeredgewidth=0.1)
    if(bDrawNumber):
    #画插值
        #x2,y = m(xnew,ynew)
        #画站点
        if(bDrawPoint):
            #m.scatter(x,y,s=4,lw=0)
            #cnfont = dclim.GetCnFont()
            #plt.plot(x,y,'o')
            #x2,y2 = m(x-0.3,x+0.04)
            pass
            #m.scatter(x,y,c=z,s=4,lw=0)

        for i in range(len(x)):
            #zc2 为 插值后的实况值
            #print(x[i],y[i],z[i])
            #plt.text(x2[i],y2[i],'%5.1f'%zc2[i],fontsize=9)
            plt.text(x[i]-10000,y[i]-13000.,'%d'%z[i],fontsize=11,color='red')
            #plt.text(xt1[i],yt1[i],'%5.1f'%z[i]+' ',fontsize=9)
            #plt.text(x2[i],y2[i],'%5.0f'%zc3[i]+' ',fontsize=9,fontproperties=cnfont)
            #画站名

            #x2,y2 = m(xnew-0.1,ynew-0.1)
            #for i in range(len(x)):
            #    str1 = unicode(staname[i],'gbk')
            #    plt.text(x2[i],y2[i],str1,fontsize=6,fontproperties=cnfont)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    import dclimate as dclim
    cnfont = dclim.GetCnFont2(12)
    plt.title(Title.decode('gb2312'),fontproperties=cnfont)


    print('shpfile_border=',shpfile_border)

    #sys.exit(0)
    b_show_ss=False

    import re
    p = re.compile('china')
    match = p.search(shpfile_border)
    if(match):
        #b_show_ss=False
        b_show_ss=True

    #b_show_ss=False
    if(b_show_ss):

        mytic()
        ax2 = fig.add_axes([0.5866,0.149,0.2,0.2])
        m2 = Basemap(projection='cyl',llcrnrlat=4,urcrnrlat=25, \
                     llcrnrlon=107,urcrnrlon=122,resolution='h',ax=ax2)
        #sf = shapefile.Reader(r"spatialdat\Export_Output_12")
        mytoc('ee')
        southsea_file = os.path.join(Spatial_Data,"Export_Output_12")

        print(southsea_file)
        #import dfunc as df
        #df.draw_map_lines(m2,southsea_file,linewidth=0.5)
        mytic()
        m2.readshapefile(southsea_file,'',drawbounds=True,color='k',linewidth=0.5)
        #m.readshapefile(southsea_file,drawbounds=True,color='k',linewidth=0.5)
        mytoc('Draw South Sea')

    #plt.show()
    #plt.colorbar()
    #plt.savefig('03.png',dpi=150)
    #print(imgfile)
    print(imgfile)
    #sys.exit(0)
    #imgfile='a.png'
    plt.savefig(imgfile,dpi=150)
    import dfunc as df
    df.CutPicWhiteBorder(imgfile)
    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    if(bShowImage):
        plt.show()
    plt.close(fig)
    print('绘图结束!!!!!!!!!!')
    mytoc('绘图所耗时间')