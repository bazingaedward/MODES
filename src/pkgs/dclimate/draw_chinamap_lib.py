# -*- coding: gb2312 -*-
from __future__ import print_function

#from osgeo import  ogr
import numpy as np
import time as getsystime
from mpl_toolkits.basemap import Basemap,shapefile
import matplotlib.pyplot as plt
import dgriddata as dgrid
import dfunc as df
from dfunc import mytic,mytoc
import sys,os
import time
from __init__ import Spatial_Data,Station_Data,Level_Path


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


#------------------------------------------------------------------------------
#装饰器测试
#dgrid.dectime2('main function')
def Draw_China_Split(filename,title1,b_show_ss=0,levfile='',b_showimg=0,imgfile='tmp1.png',b_cnfont=1,cluster_num=4,projtype=1):
    '''
    Setup: Generate data...
    '''
    nx, ny = 400*1.5,480*1.5
    #RegionR = np.loadtxt('1971_2010_45_12_1_TA.txt')
    RegionR = np.loadtxt(filename)
    Region = RegionR[1:,:]
    shapeR = np.shape(Region)
    
    if(shapeR[1]>3):
        x,y,z = Region[:,1],Region[:,2],Region[:,3]

    if(3==shapeR[1]):
        x,y,z = Region[:,0],Region[:,1],Region[:,2]


    #
    xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax = 100.0,134.7,16.2,30.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)

    #离散点插值到网格
    mytic()
    if np.size(x)>350 :
        #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='Invdisttree')#Invdisttree
        zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#Invdisttree
    else:
        #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf2')#Invdisttree
        #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='griddata')#Invdisttree
        zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#Invdisttree
        zi.round()
    mytoc(u'离散点插值到网格')
    #重要

    #平滑插值
    np.savetxt('Z.txt',zi,fmt='%02d')

    if(0):
        mytic()
        zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=2)
        zi=zi.round()
        #print(zi.ravel())
        #sys.exit(0)
        mytoc(u'扩展矩阵插值: ')
        #sys.exit(0)

    #获取mask矩阵

    mytic()
    grid1 = dgrid.build_inside_mask_array(\
        os.path.join(Spatial_Data,r"china_province"),x1,y1) #shapes
    #grid1,shapes = dgrid.build_inside_mask_array(r"spatialdat\CJ_BOUND",x1,y1)
    mytoc(u'mask非绘图区域')
    zi[np.logical_not(grid1)]=np.NaN

    #-----------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])

    #lcc 蓝波托投影
    if(1==projtype):
        m = Basemap(llcrnrlon=86,llcrnrlat=13.5,urcrnrlon=140,urcrnrlat=51,\
            projection='lcc',lon_0=110,lat_0=30.0,lat_1=60.0,resolution='l') #resolution=None
        m.drawparallels(np.arange(20,71,10),labels=[0,1,0,1],linewidth=0.2, fontsize=8)
        m.drawmeridians(np.arange(80,135,10),labels=[0,1,0,1,0,1],linewidth=0.2, fontsize=8)

    #Equidistant Cylindrical Projection 等距圆柱投影 CYL
    if(2==projtype):
        m = Basemap(projection='cyl',llcrnrlat=17,urcrnrlat=55, \
                    llcrnrlon=72,urcrnrlon=136,resolution='l')
        m.drawparallels(np.arange(20,75,5),labels=[0,1,0,1],linewidth=0.2, fontsize=8)
        m.drawmeridians(np.arange(80,135,5),labels=[0,1,0,1,0,1],linewidth=0.2, fontsize=8)


#resolution == 'c':
    #resolution == 'l':
    #resolution == 'i':
    #resolution == 'h':
    #resolution == 'f':


    m.ax=ax1

    m.drawcoastlines(linewidth=0.5)

    #m.drawcoastlines()
    #m.drawcountries(linewidth=0.2)
#    m.maskoceans()
    m.drawmapboundary()
    
    # mytic()
    # dgrid.draw_map_lines(m,'spatialdat/CJ_BOUND.shp')#os.path.join(Spatial_Data,"china_province")
    # mytoc(u'画省界')
    #
    # mytic()
    # dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"huanghe_ganliu"),color='b',linewidth=1.0)
    # mytoc(u'画黄河干流: ')
    #
    # mytic()
    # dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_LVL1_RIVER"),color='b',linewidth=1.0)
    # mytoc(u'画长江干流: ')
    # mytic()

    #mytic()
    #dgrid.draw_map_lines(m,r"spatialdat\CJ_BOUND",color='k',linewidth=1.0)
    #dgrid.draw_map_lines(m,r"spatialdat\CJ_BOUND",color='g',linewidth=0.50)
    #mytoc(u'画长江')

    mytic()
    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
    #maplev = np.loadtxt('maplev_TA.txt')


    if( os.path.isfile(levfile)):
        #maplev = np.loadtxt('maplev_RAP.lev')
        maplev = np.loadtxt(levfile)
        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)
        cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev)#,extend='both')#plt.cm.jet,
        CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
    else:
        #---------------------------------------------------
        #cax = m.contourf(xi,yi,zi,cmap=plt.cm.gist_ncar)
        #CS = m.contour(xi,yi,zi,linewidths=0.5,colors='k')
        #---------------------------------------------------
        lev = np.arange(1,cluster_num+1)+0.5
        '''
        for ii in range(1,12):
            xi2 = np.where(zi==ii,xi,np.nan)
            yi2 = np.where(zi==ii,yi,np.nan)
            zi2 = np.where(zi==ii,zi,np.nan)

            #cax = m.contourf(xi2,yi2,zi2,cmap=plt.cm.spectral,levels=lev)#,extend='both')#plt.cm.jet,
            #CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
            CS = m.contour(xi2,yi2,zi2,levels=lev,linewidths=.6,colors='k')
            #CS = m.contour(xi2,yi2,zi2,contours=1)
            #cs = m.pcolor(xi2,yi2,zi2, norm=plt.Normalize(0,17))
        #plt.colorbar(cax,shrink=0.6)
        '''
        #---------------------------------------------------
        #im1 = m.pcolor(xi,yi,zi,shading='flat',cmap=plt.cm.jet, norm=plt.Normalize(0,17))
        #cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
        #m.pcolor(xi,yi,zi,cmap=plt.cm.spectral, norm=plt.Normalize(0,17))
        cmap = plt.get_cmap('jet',cluster_num)
        cmap.set_bad(color='w',alpha=1)
        zi = np.ma.masked_invalid(zi)
        iml =m.pcolormesh(xi,yi,zi,cmap=cmap, norm=plt.Normalize(1,cluster_num+1))
        cb = m.colorbar(iml,"bottom", size="2%", pad="1%")
        cb.set_ticks(np.arange(1,cluster_num+1))
        #m.imshow(zi,interpolation="none")#interpolation="nearest")

        #---------------------------------------------------
        #m.imshow(zi, interpolation='nearest')
        #m.pcolormesh(xi,yi,zi)

    #l, b, w, h = pos.bounds
    #cax = plt.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
    #plt.colorbar(drawedges=True, cax=cax)
    ##plt.colorbar(cax,shrink=0.6)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
    #plt.colorbar(ax=ax1)



    ###plt.clabel(CS, fmt='%4.1f',inline=0.5, fontsize=8)  #inline 为等值线标注值周围的间隔

    x5,y5 = m(x,y)  #变换为proj.4

    #m.plot(x,y,c=z,s=10,markeredgewidth=0.1)

    m.plot(x5,y5,'bo',markersize=2)
    #m.scatter(x,y,s=4,lw=0)

    for ii in range(len(x)):
        plt.text(x5[ii],y5[ii],'%d'%int(z[ii]))#,fontsize=9)


    #m.plot(x,y,'o',markeredgewidth=0.1)
    import pkgs.dclimate as dclim

    if b_cnfont :

        cnfont = dclim.GetCnFont()
        #plt.title(ptitle,y=1.075, fontproperties=cnfont)
        plt.title(title1, fontproperties=cnfont,size=20)
    else:
        plt.title(title1,size=20)


    #麦卡托投影哦
    pos = ax1.get_position()
    #print(pos)
    mytoc(u'正式绘图')
    if(b_show_ss):
        ax2 = fig.add_axes([0.0533,0.1393,0.2,0.2])
        m2 = Basemap(projection='cyl',llcrnrlat=4,urcrnrlat=25, \
                 llcrnrlon=107,urcrnrlon=122,resolution='h',ax=ax2)
        #sf = shapefile.Reader(r"spatialdat\Export_Output_12")

        mytic()
        df.draw_map_lines(m2,os.path.join(Spatial_Data,"Export_Output_12"),linewidth=0.5)
        mytoc(u'画南海')

    #plt.colorbar()
    plt.savefig(imgfile,dbi=200)
    df.CutPicWhiteBorder(imgfile)

    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    if(b_showimg):
        plt.show()
    mytoc(u'绘图所耗时间')

#------------------------------------------------------------------------------


def Draw_China_Map(Region,title1,levfile='',b_showimg=0,imgfile='tmp1.png',b_show_ss=0,b_show_site=0,b_cnfont=1):
    '''
    Setup: Generate data...
    '''
    nx, ny = 100,120
    #RegionR = np.loadtxt('1971_2010_45_12_1_TA.txt')
    #RegionR = np.loadtxt(filename)
    #Region = RegionR[1:,:]
    #shapeR = np.shape(Region)

    #if(shapeR[1]>3):
    #    x,y,z = Region[:,1],Region[:,2],Region[:,3]

    #if(3==shapeR[1]):
    x,y,z = Region[:,0],Region[:,1],Region[:,2]

    #
    xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax = 100.0,134.7,16.2,30.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)

    #离散点插值到网格
    # mytic()
    # if np.size(x)>350 :
    #     #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='Invdisttree')#Invdisttree
    #     zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='griddata')#Invdisttree
    # else:
    #     zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='Invdisttree')#Invdisttree
    #     #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='griddata')#Invdisttree
    #     #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#Invdisttree
    # mytoc(u'离散点插值到网格')
    # #重要

    #平滑插值

    if(0):
        mytic()
        zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=6)
        #print(zi.ravel())
        #sys.exit(0)
        mytoc(u'扩展矩阵插值: ')
        #sys.exit(0)

    #获取mask矩阵

    # mytic()
    # grid1 = dgrid.build_inside_mask_array( \
    #     os.path.join(Spatial_Data,r"china_province"),x1,y1) #shapes
    # grid1,shapes = dgrid.build_inside_mask_array(r"spatialdat\CJ_BOUND",x1,y1)
    # mytoc(u'mask非绘图区域')
    # zi[np.logical_not(grid1)]=np.NaN

    #-----------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #原始的中国区域
    #m = Basemap(llcrnrlon=86,llcrnrlat=13.5,urcrnrlon=140,urcrnrlat=51, \
    #            projection='lcc',lon_0=110,lat_0=30.0,lat_1=60.0,resolution='l') #resolution=None

    #新的调试
    m = Basemap(llcrnrlon=86,llcrnrlat=13.5,urcrnrlon=146.5,urcrnrlat=51, \
            projection='lcc',lon_0=110,lat_0=30.0,lat_1=60.0,resolution='l') #resolution=None

    #resolution == 'c':
    #resolution == 'l':
    #resolution == 'i':
    #resolution == 'h':
    #resolution == 'f':


    m.ax=ax1
    m.drawparallels(np.arange(20,71,10),labels=[0,1,0,1],linewidth=0.2, fontsize=8)
    m.drawmeridians(np.arange(80,131,10),labels=[0,1,0,1,0,1],linewidth=0.2, fontsize=8)
    m.drawcoastlines(linewidth=0.2)
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"china_province"))
    mytoc(u'画省界')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"huanghe_ganliu"),color='b',linewidth=1.0)
    mytoc(u'画黄河干流: ')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_LVL1_RIVER"),color='b',linewidth=1.0)
    mytoc(u'画长江干流: ')
    mytic()
    plt.show()
    #mytic()
    #dgrid.draw_map_lines(m,r"spatialdat\CJ_BOUND",color='k',linewidth=1.0)
    #dgrid.draw_map_lines(m,r"spatialdat\CJ_BOUND",color='g',linewidth=0.50)
    #mytoc(u'画长江')

    mytic()
    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
    #maplev = np.loadtxt('maplev_TA.txt')

    print('levfile=',levfile)
    if( os.path.isfile(levfile)):
        #maplev = np.loadtxt('maplev_RAP.lev')
        maplev = np.loadtxt(levfile)
        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)
        #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet, #max
        CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
    else:
        #---------------------------------------------------
        #cax = m.contourf(xi,yi,zi,cmap=plt.cm.gist_ncar)
        #CS = m.contour(xi,yi,zi,linewidths=0.5,colors='k')
        #---------------------------------------------------
        lev = np.arange(-1,17)+0.5
        #print('cccccccccccccccccccccc')
        #print(xi.shape,yi.shape,zi.shape)
        #print(zi)

        cax = m.contourf(xi,yi,zi,cmap=plt.cm.spectral)#,extend='both')#plt.cm.jet, ,levels=lev
        #CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        CS = m.contour(xi,yi,zi,linewidths=.6,colors='k')  #,levels=lev
        #---------------------------------------------------
        #im1 = m.pcolor(xi,yi,zi,shading='flat',cmap=plt.cm.jet, norm=plt.Normalize(0,17))
        #cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
        #m.pcolor(xi,yi,zi)#,cmap=plt.cm.spectral)
        #m.imshow(zi)


        #---------------------------------------------------
        #m.imshow(zi, interpolation='nearest')
        #m.pcolormesh(xi,yi,zi)

    #l, b, w, h = pos.bounds
    #cax = plt.axes([l+w+0.075, b, 0.05, h]) # setup colorbar axes
    #plt.colorbar(drawedges=True, cax=cax)
    plt.colorbar(cax,shrink=0.6)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
    #plt.colorbar(ax=ax1)



    plt.clabel(CS, fmt='%4.1f',inline=1,fontsize=8)  #inline 为等值线标注值周围的间隔

    #print(x,y)
    x,y = m(x,y)  #变换为proj.4
    #print(x,y)
    #sys.exit(0)
    #m.plot(x,y,c=z,s=10,markeredgewidth=0.1)

    import dclimate as dclim
    cnfont = dclim.GetCnFont()
    if(b_show_site):
        m.scatter(x,y,s=4,lw=0)
        for ii in range(len(x)):
            #plt.text(x[ii],y[ii],'%f'%z[ii],fontsize=5)
            str1 = '%d'%z[ii];
            #print(str1)
            #plt.text(x[ii],y[ii],'1',fontsize=8)
            if b_cnfont:
                plt.text(x[ii],y[ii],str1,fontsize=6,fontproperties=cnfont)
            else:
                plt.text(x[ii],y[ii],str1,fontsize=4)


    #m.plot(x,y,'o',markeredgewidth=0.1)

    #plt.title(ptitle,y=1.075, fontproperties=cnfont)


    if b_cnfont :

        cnfont = dclim.GetCnFont()
        #plt.title(ptitle,y=1.075, fontproperties=cnfont)
        plt.title(title1, fontproperties=cnfont,size=16)
    else:
        plt.title(title1,size=16)


    #麦卡托投影哦
    pos = ax1.get_position()
    #print(pos)
    mytoc(u'正式绘图')

    if(b_show_ss):
        ax2 = fig.add_axes([0.5866,0.1482,0.2,0.2])
        m2 = Basemap(projection='cyl',llcrnrlat=4,urcrnrlat=25, \
                     llcrnrlon=107,urcrnrlon=122,resolution='h',ax=ax2)
        #sf = shapefile.Reader(r"spatialdat\Export_Output_12")
        mytic()
        southsea_file = os.path.join(Spatial_Data,"Export_Output_12")
        print(southsea_file)
        df.draw_map_lines(m2,southsea_file,linewidth=0.5)
        mytoc(u'画南海')
        #sys.exit(0)


    if(0):
        ax2 = fig.add_axes([0.0533,0.1393,0.2,0.2])
        m2 = Basemap(projection='cyl',llcrnrlat=4,urcrnrlat=25, \
                     llcrnrlon=107,urcrnrlon=122,resolution='h',ax=ax2)
        #sf = shapefile.Reader(r"spatialdat\Export_Output_12")
        mytic()
        dgrid.draw_map_lines(m2,os.path.join(Spatial_Data,"Export_Output_12"),linewidth=0.5)
        mytoc(u'画南海')

    #plt.colorbar()
    print(imgfile)
    plt.savefig(imgfile)
    #plt.savefig('tmp1.pdf')
    df.CutPicWhiteBorder(imgfile)

    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    if(b_showimg):
        plt.show()
    plt.close()
    mytoc(u'绘图所耗时间')

#------------------------------------------------------------------------------
def draw_cj_map(Region,Title1,Levfile='',b_showimg=0,imgfile='tmp1.png',projtype=1,grid_type='line_rbf'):
    '''
    Setup: Generate data...

    test command:
    RegionR = np.loadtxt('out.txt')
    Region = RegionR[1:,:]
    draw_cj_map(Region,Title1='中午测试',Levfile='',b_showimg=0,imgfile='tmp1.png')

    '''

    #(np.array(X1),np.array(Y1),ST , np.array(AVG_Val),np.array(STID) )
    xs,ys,staname,_,_=GetStaInfo(os.path.join(Station_Data,"cj_sta.txt"))
    print(staname)

    mytic()
    #Title1 = 'Title'
    #LevelFile = ''
    #nx, ny = 150,120
    nx, ny = 400,200
    #RegionR = np.loadtxt('1971_2010_45_12_1_TA.txt')

    ##RegionR = np.loadtxt('out.txt')
    ##Region = RegionR#[1:,:]
    #print(stats._support.unique(Region))
    ##shapeR = np.shape(Region)

    #print(shapeR[1])
    #sys.exit(0)
    #if(shapeR[1]>3):
    #    x,y,z = Region[:,1],Region[:,2],Region[:,3]
    #if(3==shapeR[1]):
    #if(1):
    x,y,z = Region[:,0],Region[:,1],Region[:,2]
    #
    #xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax =108,116.5,29,34
    xmin,xmax,ymin,ymax = 90,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)

    #离散点插值到网格
    #if('line_rbf2'==grid_type):
    zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func=grid_type)

    #if('Invdisttree'==grid_type):
    #    zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='Invdisttree')

    #if('line_rbf'==grid_type):
    #    zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf')

    #if('nearest'==grid_type):
    #    zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')
    #    if('nearest'==func):
    #    if('griddata'==func):
    #    if('kriging'==func):
    #    if('scipy_idw'==func):
    #    if('line_rbf'==func):
    #    if('line_rbf2'==func):
    #    if('Invdisttree'==func):
    #    if('nat_grid'==func):

    '''
    if(1):
        mytic()
        if np.size(x)>500 :
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='griddata')#Invdisttree
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf')#Invdisttree
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#Invdisttree
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf2')#Invdisttree
        else:
            #zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')#Invdisttree
            zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='line_rbf')#scipy_idw
            #line_rbf line_rbf2
        #mytoc(u'离散点插值到网格')
        mytoc(u'插值')
    '''

    #平滑插值
    print(zi.shape,x1.shape,y1.shape)
    #sys.exit(0)
    mytic()
    zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=2)
    #mytoc(u'扩展矩阵插值: ')
    mytoc(u'扩展: ')
    #sys.exit(0)

    #获取mask矩阵

    mytic()
    #grid1,shapes = df.build_inside_mask_array(r"spatialdat\china_province",x1,y1)
    #grid1,shapes = df.build_inside_mask_array(r"spatialdata\hbsj-mian",x1,y1)
    grid1 = dgrid.build_inside_mask_array(os.path.join(Spatial_Data,"CJ_BOUND"),x1,y1)
    mytoc(u'mask')#非绘图区域
    zi[np.logical_not(grid1)]=np.NaN

    #-----------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    #兰伯拖投影 lambert 投影
    if(1==projtype):
        m = Basemap(llcrnrlon=92.2,llcrnrlat=23.0,urcrnrlon=123.7,urcrnrlat=35.0, \
                projection='lcc',lon_0=105,lat_0=30.0,lat_1=60.0)
    #90,122.5,24.0,36.0
    #Equidistant Cylindrical Projection 等距圆柱投影 CYL
    if(2==projtype):
        m = Basemap(projection='cyl',llcrnrlat=24,urcrnrlat=36.4, \
                    llcrnrlon=90,urcrnrlon=122.7,resolution='l')
    m.ax=ax1
    m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    #m.drawcoastlines(linewidth=0.2)  #画海岸线
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()


    mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Province_CLIP",color='k',linewidth=0.5,debug=1)
    #m.readshapefile('CJ_GIS_INFO\CJ_Province_CLIP','',drawbounds=True,color='#0000ff',linewidth=0.2)
    m.readshapefile(os.path.join(Spatial_Data,'china_province'),'',drawbounds=True,color='#888888',linewidth=0.2)
    mytoc(u'1画流域内省界:');# ')

    mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    #m.readshapefile(os.path.join(Spatial_Data,'CJ_Basin2_Line'),'',drawbounds=True,color='r',linewidth=0.2)
    m.readshapefile(os.path.join(Spatial_Data,'upstreamCD'),'',drawbounds=True,color='r',linewidth=0.2)
    #upstreamCD
    mytoc(u'2画流域界:')# ')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_LVL1_RIVER"),color='b',linewidth=1.0)
    mytoc(u'3画长江干流: ')#')

    mytic()
    #dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"sxrivers"),color='b',linewidth=0.4)
    mytoc(u'3-1画长江干流: ')#')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_BOUND"),color='k',linewidth=1.0)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_BOUND"),color='g',linewidth=0.50)
    mytoc(u'4画省界')

    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]


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
        cax = m.contourf(xi,yi,zi)
        #CS = m.contour(xi,yi,zi,levels=lev,linewidths=0.5,colors='k')
        CS = m.contour(xi,yi,zi,linewidths=.6,colors='k')  #,levels=lev
        plt.colorbar(cax,shrink=0.3)#orientation='vertical',shrink=0.6,ticks=lev)#,drawedges=True,
        plt.clabel(CS, fmt='%4.1f',inline=0.5, fontsize=8)  #inline 为等值线标注值周围的间隔



    x5,y5 = m(x,y)
    x,y = m(x,y)  #变换为proj.4

    #m.plot(x,y,c=z,s=10,markeredgewidth=0.1)
    #m.scatter(x,y,c=z,s=4,lw=0)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    import dclimate as dclim
    cnfont = dclim.GetCnFont()

    plt.title(Title1.decode('gbk'),fontproperties=cnfont,size=12)
    #麦卡托投影

    #################
    xsm,ysm = m(xs,ys)
    xsm2,ysm2 = m(xs-0.5,ys+0.2)
    m.plot(xsm,ysm,'o',markersize=4,markeredgewidth=0.01)
    for ii in range(len(staname)):
        str1 = unicode(staname[ii],'gbk')
        plt.text(xsm2[ii],ysm2[ii],str1,fontsize=9,fontproperties=cnfont)

        #plt.text(x2[i],y2[i],str1,fontsize=6,fontproperties=cnfont)
      #变换为proj.4

    #m.plot(x,y,c=z,s=10,markeredgewidth=0.1)

    #m.plot(x5,y5,'bo',markersize=2)

    #plt.colorbar()
    #print(Magick_Convert)
    #imgfile = 'out.png'
    plt.savefig(imgfile, dpi=120)
    df.CutPicWhiteBorder(imgfile)



    #plt.savefig('03.png',dpi=120)
    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    #plt.savefig('03.eps')
    if(b_showimg):
        plt.show()
    plt.close()

    mytoc(u'绘图所耗时间')
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def draw_cj_map_split(Region,Title1,Levfile='',b_showimg=0,imgfile='tmp1.png',cluster_num=4,projtype=1):
    '''
    Setup: Generate data...

    test command:
    RegionR = np.loadtxt('out.txt')
    Region = RegionR[1:,:]
    draw_cj_map(Region,Title1='中午测试',Levfile='',b_showimg=0,imgfile='tmp1.png')

    '''
    mytic()
    #Title1 = 'Title'
    #LevelFile = ''
    nx, ny = 70,30
    #nx, ny = 400,200
    #RegionR = np.loadtxt('1971_2010_45_12_1_TA.txt')

    ##RegionR = np.loadtxt('out.txt')
    ##Region = RegionR#[1:,:]
    #print(stats._support.unique(Region))
    ##shapeR = np.shape(Region)

    #print(shapeR[1])
    #sys.exit(0)
    #if(shapeR[1]>3):
    #    x,y,z = Region[:,1],Region[:,2],Region[:,3]
    #if(3==shapeR[1]):
    #if(1):
    x,y,z = Region[:,0],Region[:,1],Region[:,2]
    print(x)
    print(y)
    print(z)
    #
    #xmin,xmax,ymin,ymax = 73.6,134.7,16.2,54.0
    #xmin,xmax,ymin,ymax =108,116.5,29,34
    #原始信息 xmin,xmax,ymin,ymax = 90,122.5,24.0,36.0
    xmin,xmax,ymin,ymax = 98,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)

    print('lontlatbox=',x.min(),x.max(),y.min(),y.max())
    print(x1.shape,y1.shape)

    #离散点插值到网格
    if(1):
        mytic()
        #if np.size(x)>500 :
        #    zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='Invdisttree')#Invdisttree
        #else:
        zi,xi,yi = dgrid.griddata_all(x,y,z,x1,y1,func='nearest')
            #line_rbf line_rbf2 nearest
        #mytoc(u'离散点插值到网格')
        mytoc(u'插值')


    #平滑插值
    print(zi.shape,x1.shape,y1.shape)
    #sys.exit(0)
    mytic()
    #zi,xi,yi,x1,y1,nx,ny=dgrid.extened_grid(zi,x1,y1,zoom=2)
    #mytoc(u'扩展矩阵插值: ')
    mytoc(u'扩展: ')
    #sys.exit(0)

    #获取mask矩阵

    mytic()
    #grid1,shapes = df.build_inside_mask_array(r"spatialdat\china_province",x1,y1)
    #grid1,shapes = df.build_inside_mask_array(r"spatialdata\hbsj-mian",x1,y1)
    grid1 = dgrid.build_inside_mask_array(os.path.join(Spatial_Data,"CJ_BOUND"),x1,y1)
    mytoc(u'mask')#非绘图区域
    zi[np.logical_not(grid1)]=np.NaN

    #-----------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    #兰伯拖投影 lambert 投影
    if(1==projtype):
        m = Basemap(llcrnrlon=92.2,llcrnrlat=23.0,urcrnrlon=123.7,urcrnrlat=35.0, \
                    projection='lcc',lon_0=105,lat_0=30.0,lat_1=60.0)
        #90,122.5,24.0,36.0
    #Equidistant Cylindrical Projection 等距圆柱投影 CYL
    if(2==projtype):
        m = Basemap(projection='cyl',llcrnrlat=24,urcrnrlat=36.4, \
                    llcrnrlon=90,urcrnrlon=122.7,resolution='l')
    m.ax=ax1
    m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    #m.drawcoastlines(linewidth=0.2)  #画海岸线
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()

    mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Province_CLIP",color='k',linewidth=0.5,debug=1)
    #m.readshapefile('CJ_GIS_INFO\CJ_Province_CLIP','',drawbounds=True,color='#0000ff',linewidth=0.2)
    m.readshapefile(os.path.join(Spatial_Data,'china_province'),'',drawbounds=True,color='#888888',linewidth=0.2)
    mytoc(u'1画流域内省界:');# ')

    mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    m.readshapefile(os.path.join(Spatial_Data,'CJ_Basin2_Line'),'',drawbounds=True,color='r',linewidth=0.2)
    mytoc(u'2画流域界:')# ')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_LVL1_RIVER"),color='b',linewidth=1.0)
    mytoc(u'3画长江干流: ')#')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"sxrivers"),color='b',linewidth=0.4)
    mytoc(u'3-1画长江干流: ')#')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_BOUND"),color='k',linewidth=1.0)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_BOUND"),color='g',linewidth=0.50)
    mytoc(u'4画省界')


    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
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
        #cmap = plt.get_cmap('jet',cluster_num)
        #cmap.set_bad(color='w',alpha=1)
        '''
        zi = np.ma.masked_invalid(zi)
        print(zi)
        iml =m.pcolormesh(xi,yi,zi,edgecolors='w', linewidths=0.01)#,cmap=cmap, norm=plt.Normalize(1,cluster_num+1))
            cb = m.colorbar(iml,"right", size="2%", pad="1%")
        '''
        cmap = plt.get_cmap('jet',cluster_num)
        cmap.set_bad(color='w',alpha=1)
        zi = np.ma.masked_invalid(zi)
        iml =m.pcolormesh(xi,yi,zi,cmap=cmap, norm=plt.Normalize(1,cluster_num+1))
        cb = m.colorbar(iml,"bottom", size="2%", pad="1%")
        cb.set_ticks(np.arange(1,cluster_num+1))



        #cb.set_ticks(np.arange(1,cluster_num+1))

    x5,y5 = m(x,y)  #变换为proj.4

    m.plot(x5,y5,'bo',markersize=2)
    #m.plot(x,y,c=z,s=10,markeredgewidth=0.1)
    #m.scatter(x,y,c=z,s=4,lw=0)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    import dclimate as dclim
    cnfont = dclim.GetCnFont()

    plt.title(Title1.decode('gbk'),fontproperties=cnfont,size=12)

    #麦卡托投影哦

    #plt.colorbar()
    #print(Magick_Convert)
    #imgfile = 'out.png'
    plt.savefig(imgfile, dpi=120)
    df.CutPicWhiteBorder(imgfile)



    #plt.savefig('03.png',dpi=120)
    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    #plt.savefig('03.eps')
    if(b_showimg):
        plt.show()

    mytoc(u'绘图所耗时间')
#------------------------------------------------------------------------------

def draw_cj_map_split2(zi,Title1,Levfile='',b_showimg=0,imgfile='tmp1.png',cluster_num=4,projtype=1):
    '''
    Setup: Generate data...

    test command:
    RegionR = np.loadtxt('out.txt')
    Region = RegionR[1:,:]
    draw_cj_map(Region,Title1='中午测试',Levfile='',b_showimg=0,imgfile='tmp1.png')

    '''
    nx, ny = 70,30
    xmin,xmax,ymin,ymax = 98,122.5,24.0,36.0

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)
    xi, yi = np.meshgrid(x1, y1)

    #-----------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=108,llcrnrlat=29,urcrnrlon=116.9,urcrnrlat=33.6,\
    #兰伯拖投影 lambert 投影
    if(1==projtype):
        m = Basemap(llcrnrlon=92.2,llcrnrlat=23.0,urcrnrlon=123.7,urcrnrlat=35.0, \
                    projection='lcc',lon_0=105,lat_0=30.0,lat_1=60.0)
        #90,122.5,24.0,36.0
    #Equidistant Cylindrical Projection 等距圆柱投影 CYL
    if(2==projtype):
        m = Basemap(projection='cyl',llcrnrlat=24,urcrnrlat=36.4, \
                    llcrnrlon=90,urcrnrlon=122.7,resolution='l')
    m.ax=ax1
    m.drawparallels(np.arange(20,71,5),labels=[1,0,0,0,],linewidth=0.3, fontsize=10)
    m.drawmeridians(np.arange(80,131,5),labels=[0,0,0,1],linewidth=0.3, fontsize=10)
    #m.drawcoastlines(linewidth=0.2)  #画海岸线
    #m.drawcountries(linewidth=0.2)
    #m.drawmapboundary()


    mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Province_CLIP",color='k',linewidth=0.5,debug=1)
    #m.readshapefile('CJ_GIS_INFO\CJ_Province_CLIP','',drawbounds=True,color='#0000ff',linewidth=0.2)
    m.readshapefile(os.path.join(Spatial_Data,'china_province'),'',drawbounds=True,color='#888888',linewidth=0.2)
    mytoc(u'1画流域内省界:');# ')

    mytic()
    #df.draw_map_lines(m,r"CJ_GIS_INFO\CJ_Basin2_Line",color='r',linewidth=0.2)
    m.readshapefile(os.path.join(Spatial_Data,'CJ_Basin2_Line'),'',drawbounds=True,color='r',linewidth=0.2)
    mytoc(u'2画流域界:')# ')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,'CJ_LVL1_RIVER'),color='b',linewidth=1.0)
    mytoc(u'3画长江干流: ')#')

    mytic()
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,'CJ_BOUND'),color='k',linewidth=1.0)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,'CJ_BOUND'),color='g',linewidth=0.50)
    mytoc(u'4画省界')

    xi, yi = m(xi,yi)
    #lev = [-150, -100, -50, -20, 0 ,20, 50, 100, 200]
    #lev = [-3.,-2., -1., -0.5,  0. ,0.5, 1., 2., 3.]
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
        #cmap = plt.get_cmap('jet',cluster_num)
        #cmap.set_bad(color='w',alpha=1)
        '''
        zi = np.ma.masked_invalid(zi)
        print(zi)
        iml =m.pcolormesh(xi,yi,zi,edgecolors='w', linewidths=0.01)#,cmap=cmap, norm=plt.Normalize(1,cluster_num+1))
            cb = m.colorbar(iml,"right", size="2%", pad="1%")
        '''
        cmap = plt.get_cmap('jet',cluster_num)
        cmap.set_bad(color='w',alpha=1)
        zi = np.ma.masked_invalid(zi)
        print(zi)
        iml =m.pcolormesh(xi,yi,zi,cmap=cmap, norm=plt.Normalize(1,cluster_num+1))
        cb = m.colorbar(iml,"bottom", size="2%", pad="1%")
        cb.set_ticks(np.arange(1,cluster_num+1))



        #cb.set_ticks(np.arange(1,cluster_num+1))

    #m.plot(x,y,c=z,s=10,markeredgewidth=0.1)
    #m.scatter(x,y,c=z,s=4,lw=0)

    #m.plot(x,y,'o',markeredgewidth=0.1)
    import dclimate as dclim
    cnfont = dclim.GetCnFont()

    plt.title(Title1.decode('gbk'),fontproperties=cnfont,size=12)

    #麦卡托投影哦

    #plt.colorbar()
    #print(Magick_Convert)
    #imgfile = 'out.png'
    plt.savefig(imgfile, dpi=120)
    df.CutPicWhiteBorder(imgfile)



    #plt.savefig('03.png',dpi=120)
    #plt.savefig('03.pdf')
    #plt.savefig('03.ps')
    #plt.savefig('03.svg')
    #plt.savefig('03.eps')
    if(b_showimg):
        plt.show()

    mytoc(u'绘图所耗时间')
#------------------------------------------------------------------------------

def plot(x,y,z,grid):
    plt.figure()
    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
    #plt.imshow(grid, extent=(-0.1,1.1,1.1,-0.1))
    plt.hold(True)
    plt.scatter(x,y,c=z)
    plt.colorbar()

def main():
    mytic()
    Draw_China_Map('out.txt',u'测试',1,'maplev_RAP.lev')
    mytoc(u'总的时间消耗 :')

#------------------------------------------------------------------------------
#获取站点信息
#
def GetStaInfo(FileName):
    X1 = []
    Y1 = []
    ST = []
    STID=[]
    AVG_Val = []
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
        AVG_Val.append(float(l1[4]))

    f.close
    return(np.array(X1),np.array(Y1),ST , np.array(AVG_Val),np.array(STID) )

#------------------------------------------------------------------------------
if __name__ == '__main__':
    '''
    buffer = ''
    class Logger:
        def write(self, s):
            global buffer
            buffer += s
    
    mylogger = Logger()

    stdout_ = sys.stdout # backup reference to the old stdout.
    #sys.stdout = mylogger
    main()
    
    sys.stdout = stdout_
    logfile = open('log.txt','w')
    #print(buffer,file=logfile)
    print(type(buffer))
    logfile.writelines(buffer.encode('cp936'))
    logfile.close()
    '''
    #RegionR = np.loadtxt(u'E:\BDATA3\py307_cluster\App03\out.txt')
    #Region = RegionR[1:,:]
    #draw_cj_map(Region,Title1='中文测试',Levfile='',b_showimg=1,imgfile=u'E:\BDATA3\py307_cluster\App03\tmp1.png')




