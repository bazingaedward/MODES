# -*- coding: gb2312 -*-
from __future__ import print_function
#from osgeo import  ogr
import numpy as np
import time as getsystime
from scipy import interpolate
from mpl_toolkits.basemap import Basemap,shapefile
import matplotlib.pyplot as plt
import dgriddata as dgriddata
import dfunc as df
import dclimate as dclim
from dfunc import mytic,mytoc
import os,sys,datetime
import time
import ConfigParser



#------------------------------------------------------------------------------
df.dectime2('main function')
def GetDict():
    dict1={}
    from clim_modes import  getDClimDiagini
    dic1 = getDClimDiagini()
    MODES_PATH = os.environ.get('MODES')
    MODES_INI_File = os.path.join( MODES_PATH,'MODES.ini')
    MODES_INI4Pred = os.path.join( MODES_PATH,'MODES4Pred.ini')
    print(MODES_INI_File)
    x1,x2,y1,y2,shapefile1 = getXYArea(MODES_INI_File)

    dict1['xmin'],dict1['xmax']=x1,x2
    dict1['ymin'],dict1['ymax']=y1,y2

    bx1,bx2,by1,by2=getXYArea2(MODES_INI_File)

    dict1['xmin2'],dict1['xmax2']=bx1,bx2
    dict1['ymin2'],dict1['ymax2']=by1,by2

    shapefile1=shapefile1.replace('${INITDIR}',MODES_PATH)
    dict1['shapefile']=shapefile1
    dict1['LevelFile']=getLevelFile(MODES_INI4Pred)
    dict1['Title']=getiniTitle(MODES_INI4Pred)
    dict1['ProjType']=getRegionProjType(MODES_INI_File)

    dict1['bShowImage']=dic1['bShowImage']
    dict1['bDrawNumber']=dic1['bDrawNumber']
    #dict1['FileStrHead']=dic1['FileStrHead']
    (Year1,Month1,Interval1,Count1,PredObjType1)=GetPredYearMonth(MODES_INI4Pred)
    dict1['Year']=Year1
    dict1['Month']=Month1
    dict1['Interval']=Interval1
    dict1['Count']=Count1
    dict1['PredObjType']=PredObjType1
    return dict1


    #DrawMapMain(dict1,inputfile=inputfile1,imgfile=dic1['FileStrHead']+'_Pred.png')

def DrawMapMain(dict1,inputfile='out.txt',imgfile='out.png',bShowImage=True):
    '''
    Setup: Generate data...
    '''
    xmin,xmax,ymin,ymax = dict1['xmin'],dict1['xmax'],dict1['ymin'],dict1['ymax']
    shpfile1=dict1['shapefile']

    LevelFile = dict1['LevelFile']
    print('LevelFile=',LevelFile)
    #sys.exit(0)
    LonCenter = (xmin+xmax)/2.0
    Title1 = dict1['Title']

    print(shpfile1)
    #sys.exit(0)
    nx, ny = 200,200

    if(not os.path.isfile(inputfile)):
        print('输入文件%s不存在，请检查！'%(inputfile))
        sys.exit(0)
    Region = np.loadtxt(inputfile)
    #print(Region)
    x,y,z = Region[:,1],Region[:,2],Region[:,3]

    #xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax =108,116.5,29,34  //90,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)


    #离散点插值到网格
    mytic()
    #    if np.size(x)>1350 :
    #    zi,xi,yi = df.griddata_all(x,y,z,x1,y1,func='Invdisttree')#Invdisttree
    zi,xi,yi = dgriddata.griddata_all(x,y,z,x1,y1,func='line_rbf')#scipy_idw')# #line_rbf
    #zi,xi,yi = df.griddata_all(x,y,z,x1,y1,func='kriging')#scipy_idw')# #line_rbf

    mytoc('离散点插值到网格')

    #重要
    #http://hyry.dip.jp:8000/scipybook/default/file/03-scipy/scipy_interp2d.py
    mytic()

    #http://efreedom.com/Question/1-3526514/Problem-2D-Interpolation-SciPy-Non-Rectangular-Grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid

    mytic()
    zi,xi,yi,x1,y1,nx,ny=dgriddata.extened_grid(zi,x1,y1,zoom=int(2)) #
    mytoc('扩展矩阵插值: ')
    #sys.exit(0)

    #获取mask矩阵
    mytic()
    #grid1,shapes = df.build_inside_mask_array(r"spatialdat\china_province",x1,y1)
    #grid1,shapes = df.build_inside_mask_array(r"spatialdata\hbsj-mian",x1,y1)
    #a1=print(a)

    grid1 = dgriddata.build_inside_mask_array(shpfile1,x1,y1)
    mytoc('mask非绘图区域')
    zi[np.logical_not(grid1)]=np.NaN

    #-----------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    xmin2,xmax2,ymin2,ymax2 = dict1['xmin2'],dict1['xmax2'],dict1['ymin2'],dict1['ymax2']
    print(xmin2,ymin2,xmax2,ymax2)
    #sys.exit(0)
    ProjType=dict1['ProjType']

    if('lcc'==ProjType):
        m = Basemap(llcrnrlon=xmin2,llcrnrlat=ymin2,urcrnrlon=xmax2,urcrnrlat=ymax2,\
                projection='lcc',lon_0=LonCenter,lat_0=30.0,lat_1=60.0)
    if('merc'==ProjType):
        #90,122.5,24.0,36.0
        m = Basemap(projection='merc',llcrnrlat=ymin2,urcrnrlat=ymax2,\
                llcrnrlon=xmin2,urcrnrlon=xmax2,resolution='c')

    #m = Basemap(projection='merc',llcrnrlat=24,urcrnrlat=36,\
    #    llcrnrlon=90,urcrnrlon=122.5,resolution='c')
    m.ax=ax1
    m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    #m.drawcoastlines(linewidth=0.2)  #画海岸线
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()


    #mytic()
    #    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Province_CLIP",color='k',linewidth=0.5,debug=1)
    #    #m.readshapefile('CJ_GIS_INFO\CJ_Province_CLIP','',drawbounds=True,color='#0000ff',linewidth=0.2)
    #m.readshapefile('spatialdata\china_province','',drawbounds=True,color='#888888',linewidth=0.2)
    #mytoc(u'画流域内省界: ')

    #mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    #m.readshapefile('CJ_GIS_INFO\CJ_Basin2_Line','',drawbounds=True,color='r',linewidth=0.2)
    #mytoc(u'画流域界: ')

    #mytic()
    #df.draw_map_lines(m,r"spatialdata\CJ_LVL1_RIVER",color='b',linewidth=1.0)
    #mytoc(u'画长江干流: ')

    mytic()
    #df.draw_map_lines(m,shpfile1,color='k',linewidth=1.0)
    #df.draw_map_lines(m,shpfile1,color='g',linewidth=0.50)
    print(shpfile1)
    #sys.exit(0)
    m.readshapefile(shpfile1,'',drawbounds=True,color='k',linewidth=0.2)
    mytoc('画省界')

    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
    maplev = np.loadtxt(LevelFile)#'LEV\maplev_TA.LEV')

    cmap2 = maplev[:,:-1]
    cmap2 = cmap2/255.0
    lev = maplev[:,-1]
    #print(cmap2)

    cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,

    #l, b, w, h = pos.bounds
    #cax = plt.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
    #plt.colorbar(drawedges=True, cax=cax)
    plt.colorbar(cax,shrink=0.5)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
    #plt.colorbar(ax=ax1)
    CS = m.contour(xi,yi,zi,levels=lev,linewidths=1,colors='k')
    plt.clabel(CS, fmt='%4.1f',inline=0.5, fontsize=8)  #inline 为等值线标注值周围的间隔

    x,y = m(x,y)  #变换为proj.4
    #m.plot(x,y,c=z,markeredgewidth=0.1)


    if(dict1['bDrawNumber']):
    #画插值
        #x2,y = m(xnew,ynew)
        #画站点
        m.scatter(x,y,s=4,lw=0)
        #plt.plot(x,y,'o')
        #x2,y2 = m(x-0.3,x+0.04)
        m.scatter(x,y,c=z,s=4,lw=0)
        cnfont = dclim.GetCnFont()
        for i in range(len(x)):
            #zc2 为 插值后的实况值
            #print(x[i],y[i],z[i])
            #plt.text(x2[i],y2[i],'%5.1f'%zc2[i],fontsize=9)
            plt.text(x[i]-0.3,y[i]+0.04,'%5.1f'%z[i]+' ',fontsize=9)
            #plt.text(x2[i],y2[i],'%5.0f'%zc3[i]+' ',fontsize=9,fontproperties=cnfont)
    #画站名

        #x2,y2 = m(xnew-0.1,ynew-0.1)
        #for i in range(len(x)):
        #    str1 = unicode(staname[i],'gbk')
        #    plt.text(x2[i],y2[i],str1,fontsize=6,fontproperties=cnfont)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    cnfont = dclim.GetCnFont()
    plt.title(Title1.decode('gb2312'),fontproperties=cnfont)

    #plt.colorbar()
    #plt.savefig('03.png',dpi=150)
    plt.savefig(imgfile, dpi=150)
    df.CutPicWhiteBorder(imgfile)
    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    if(dict1['bShowImage']):
        plt.show()
    plt.close(fig)
    print('绘图结束!!!!!!!!!!')
    mytoc('绘图所耗时间')


def griddata_scipy_spatial(dataX,dataY,gridX,gridY,grid,maxDist=np.inf):
    from scipy import spatial

    # Assumes dataX and dataY are row vectors
    dataXY = np.vstack((dataX, dataY)).T
    dataQuadtree = spatial.KDTree(dataXY)

    xx,yy = np.meshgrid(gridX, gridY)
    xx,yy = xx.flatten, yy.flatten
    gridPoints = np.vstack((xx,yy)).T

    dists, indexes = dataQuadtree.query(gridPoints, k=1, distance_upper_bound=maxDist)
    mask = dists < maxDist
    mask = mask.reshape(grid.shape)
    grid = np.ma.masked_array(grid, mask)
    return grid

def getXYArea(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    #RegionID = config1.get("Region","RegionID")

    #Region_Section = config1.get("GLOBAL","Region_Section")
    Region_Section = config1.get("GLOBAL","Region_Section")
    Shpfile = config1.get(Region_Section,"RegionShapeFile")

    Shpfile=Shpfile.replace('.shp','')
    limitStr = config1.get(Region_Section,"RegionArea")
    list1 = limitStr.split(',');
    #print(list1)
    return(float(list1[0]),float(list1[1]),float(list1[2]),float(list1[3]),Shpfile)

def getXYArea2(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    #RegionID = config1.get("RegionSelect","RegionID")
    #Shpfile = config1.get("RegionSelect","MainShpfile")
    #Shpfile=Shpfile.replace('.shp','')
    Region_Section = config1.get("GLOBAL","Region_Section")
    #Shpfile = config1.get(Region_Section,"RegionShapeFile")
    limitStr = config1.get(Region_Section,"DrawArea")
    list1 = limitStr.split(',');
    #print(list1)
    return(float(list1[0]),float(list1[1]),float(list1[2]),float(list1[3]))

def getRegionProjType(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    #RegionID = config1.get("RegionSelect","RegionID")
    #Shpfile = config1.get("RegionSelect","MainShpfile")
    #Shpfile=Shpfile.replace('.shp','')
    Region_Section = config1.get("GLOBAL","Region_Section")
    ProjType = config1.get(Region_Section,"ProjType")
    return ProjType
	
def getLonCenter(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    RegionID = config1.get("RegionSelect","RegionID")
    LonCenter = config1.get(RegionID,"LonCenter",110)
    return(LonCenter)	
    
def getLevelFile(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    PredObjType=config1.get('FieldPara',"PredObjType")
    MODES_PATH = os.environ.get('MODES')
    #print('aaaa=',MODES_PATH)
    if(1==int(PredObjType)):
        #print('type=1')
        LevelFile= os.path.join( MODES_PATH,'LEV\maplev_RAP.LEV')

    if(2==int(PredObjType)):
        #print('type=2')
        LevelFile= os.path.join( MODES_PATH,'LEV\maplev_TA.LEV')
    #LevelFile=E:\projects08\806_MakePred\MakePred05\LEV\maplev_RAP.LEV
    #LevelFile = config1.get("RegionSelect","LevelFile")
    #print('bbbb=',LevelFile)
    return LevelFile

def getiniTitle(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    Title = config1.get('FieldPara',"Title")
    return Title

def GetPredYearMonth(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    Year1 = config1.get('FieldPara',"Year")
    Month1 = config1.get('FieldPara',"Month")
    Interval1 = config1.get('FieldPara',"Interval")
    Count1 = config1.get('FieldPara',"Count")
    PredObjType1 = config1.get('FieldPara',"PredObjType")
    return(Year1,Month1,Interval1,Count1,PredObjType1)



def GetStaInfo(FileName):
    X1 = []
    Y1 = []
    ST = []
    AVG_Val = []
    f=open(FileName)
    List1 = f.readlines()
    for txt in List1:
        #print(txt)
        txt.strip()
        l1 = txt.split()
        X1.append(float(l1[1]))
        Y1.append(float(l1[2]))
        ST.append(l1[3])
        AVG_Val.append(float(l1[4]))
        
    f.close
    return(np.array(X1),np.array(Y1),ST , np.array(AVG_Val) )
    #print(List1)
    
    

#------------------------------------------------------------------------------
def plot(x,y,z,grid):
    plt.figure()
    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
    #plt.imshow(grid, extent=(-0.1,1.1,1.1,-0.1))
    plt.hold(True)
    plt.scatter(x,y,c=z)
    plt.colorbar()

########################################################################
def GetDictbyLocalFile(MODES_PATH,MODES_INI_File,MODES_INI4Pred):
    dict1={}
    from clim_modes import  getDClimDiagini
    dic1 = getDClimDiagini()

    x1,x2,y1,y2,shapefile1 = getXYArea(MODES_INI_File)

    dict1['xmin'],dict1['xmax']=x1,x2
    dict1['ymin'],dict1['ymax']=y1,y2

    bx1,bx2,by1,by2=getXYArea2(MODES_INI_File)

    dict1['xmin2'],dict1['xmax2']=bx1,bx2
    dict1['ymin2'],dict1['ymax2']=by1,by2

    shapefile1=shapefile1.replace('${INITDIR}',MODES_PATH)
    dict1['shapefile']=shapefile1
    dict1['LevelFile']=getLevelFile(MODES_INI4Pred)
    dict1['Title']=getiniTitle(MODES_INI4Pred)
    dict1['ProjType']=getRegionProjType(MODES_INI_File)

    dict1['bShowImage']=dic1['bShowImage']
    dict1['bDrawNumber']=dic1['bDrawNumber']
    dict1['FileStrHead']=dic1['FileStrHead']
    (Year1,Month1,Interval1,Count1,PredObjType1)=GetPredYearMonth(MODES_INI4Pred)
    dict1['Year']=Year1
    dict1['Month']=Month1
    dict1['Interval']=Interval1
    dict1['Count']=Count1
    dict1['PredObjType']=PredObjType1
    return dict1



#------------------------------------------------------------------------------
if __name__ == '__main__':
    buffer = ''
    
    class Logger:
        def write(self, s):
            global buffer
            buffer += s
    
    mylogger = Logger()

    stdout_ = sys.stdout # backup reference to the old stdout.
    #sys.stdout = mylogger
    mytic()
    
    dict1 = GetDict()
    #DrawMapMain(dict1,'pred',)
    DrawMapMain(dict1,inputfile='Pred.txt',imgfile=dict1['FileStrHead']+'_Pred.png')
	
	
    mytoc(u'总的时间消耗 :')
    
    sys.stdout = stdout_
    logfile = open('log.txt','w')
    #print(buffer,file=logfile)
    print(type(buffer))
    logfile.writelines(buffer.encode('cp936'))
    logfile.close()



#select a.i_stationid,b.d_long,b.d_lat,a.c_station,cast(a.d_tt as NUMERIC(8,2))
#from sky_surface_month_avg a,sky_station_type b
#where extract(month from t_date)=2 and a.i_stationid=b.i_stationid
#and b.i_sta_type=45 and a.d_tt is not null order by a.d_tt desc
