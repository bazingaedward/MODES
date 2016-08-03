# -*- coding: cp936 -*-
'''
用于绘制面雨量分布图的延伸期预报图
2015-01-31
'''
from __future__ import print_function
import numpy as np
import dclimate.dfunc as df
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import dclimate.dclimate as dclim
import sys,os
import string
import ConfigParser
# Lambert Conformal map of lower 48 states.

def Draw_Area_Rain(FileName = 'AREA_RAIN.txt'): #AREA_RAIN #prec_real
    MODES_PATH = os.environ.get('SKY_RAIN_MONI')
    print(MODES_PATH)
    dict1=GetDict()
    print(dict1)
    #sys.exit(0)

    list1 = GetAreaInfo(FileName)
    MONI_PATH = os.environ.get('SKY_RAIN_MONI')
    #IniFileName1 =#os.path.join( MONI_PATH,'Moni4DrawMap.ini')
    IniFileName1 ='Moni4DrawMap.ini'
    Title1 = getiniTitle(IniFileName1)

    m = Basemap(llcrnrlon=dict1['ymin'],llcrnrlat=dict1['xmin'],urcrnrlon=dict1['ymax'],urcrnrlat=dict1['xmax'],
                projection='lcc',lat_1=30,lat_2=60,lon_0=dict1['lon0'])

    print(dict1['ShpFile'])
    #sys.exit(0)
    shp_info = m.readshapefile(dict1['ShpFile'],'states',drawbounds=True,color='r',linewidth=0.8)


    m.drawparallels(np.arange(20,65,5),labels=[0,1,0,0],linewidth=0.3, fontsize=8)
    m.drawmeridians(np.arange(80,121,5),labels=[0,0,0,1],linewidth=0.3, fontsize=8)

    #m.drawparallels(np.arange(20,65,10),labels=[0,0,0,0])
    #m.drawmeridians(np.arange(80,121,10),labels=[0,0,0,0])

    # choose a color for each state based on population density.
    colors={}
    statenames=[]
    cmap = plt.cm.jet # use 'hot' colormap
    vmin = 0; vmax = 450 # set range.
    print(m.states_info[0].keys())

    print('select Area Name:')
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        print(statename)
        statename = statename.decode('gb2312')
        print(statename)

        # skip DC and Puerto Rico.
        #if statename not in ['District of Columbia','Puerto Rico']:
            #pop = popdensity[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            #colors[statename] = cmap(1.-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
        #cmap
        statenames.append(statename)
    # cycle through state names, color each one.
    #sys.exit(0)
    ax = plt.gca() # get current axes instance
    i=0;
    print(m)
    #sys.exit(0)
    cnfont = dclim.GetCnFont()
    print('len=',len(statenames))
    for nshape,seg in enumerate(m.states):

        i=i+1
        print(i)
        # skip DC and Puerto Rico.
        #if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        #if statenames[nshape] in [u'岷沱江']:

        if 1:
            for L1 in list1:
                print(L1)
            #sys.exit(0)
            ValOut=0
            StrOut=''
            for L1 in list1:
                #print(L1[1].decode('gb2312'))
                #print(L1)
                if(statenames[nshape]==L1[1]):
                    ValOut = L1[2]
                    StrOut = '%5.1f'%(L1[2])
                    #print('%5.2f'%(L1[2]))
                    break
            #maplev = np.loadtxt('maplev_rain.LEV')#'LEV\maplev_TA.LEV')
            maplev = np.loadtxt(dict1['LevelFile'])#'LEV\maplev_TA.LEV')
            cmap2 = maplev[:,:-1]
            cmap2 = cmap2/255.0
            lev = maplev[:,-1]
            #maplev = np.loadtxt('maplev_rap.LEV')#'LEV\maplev_TA.LEV')
            ###if(32766==ValOut):
            ###    continue
            ##print('ValOut=',ValOut)
            #print('maplev=',maplev)
            #print(ValOut)


            if( 'ValOut' in dir() ):
                #sys.exit(0)
                color = GetAreaRainColor(maplev,ValOut)
            else:
                color = np.array([1.,1.,1.])
            #color =rgb2hex((random.uniform(0,1),0,1))
            #rgb2hex((i/42.0,i/42.0,0.3))
            #(cmap(i)[:3])
            #((i/42.0,i/42.0,0.3))
            #colors[statenames[nshape]])

            ##print(type(seg))
            ##print(seg[0])
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)

            #print(seg[:])
            #print(seg[0][:])

            xy1 = np.mean(seg,axis=0);

            #ax.text(seg[0][0],seg[0][1],statenames[nshape],fontproperties=cnfont,color='red',size=18)

            ax.text(xy1[0],xy1[1]+50000,statenames[nshape],fontproperties=cnfont,color='black',size=9)
            ax.text(xy1[0],xy1[1],StrOut,fontproperties=cnfont,color='black',size=12)

            #print(type(seg[0]),seg[0])
            #print(np.mean(seg,axis=0))
            #sys.exit(0)
            #grid2 = points_inside_poly(Points, seg)
            #print(grid2)

    # draw meridians and parallels.
    #sys.exit(0)
    if('Level01'==dict1['Level']):

        #fn1 = os.path.join(MODES_PATH,r"ShapeFile\upstream2_2\upstreamCD2")
        #df.draw_map_lines(m,fn1,color='black',linewidth=0.5)

        # #-----------------------------------------------------------------------
        fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_LVL2_RIVER")
        df.draw_map_lines(m,fn1,color='b',linewidth=.5)
        # #-----------------------------------------------------------------------
        fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_LVL3_RIVER")
        df.draw_map_lines(m,fn1,color='b',linewidth=.3)
        # #-----------------------------------------------------------------------
        fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_LVL1_RIVER_Clip")
        m.readshapefile(fn1,'',drawbounds=True,color='b',linewidth=1.2)

        #-----------------------------------------------------------------------
        fn1 = os.path.join(MODES_PATH,r"ShapeFile\ChinaProvince\china_province")
        df.draw_map_lines(m,fn1,color=[0.5,0.5,0.5],linewidth=.3)


    if('Level02'==dict1['Level']):

        fn1 = os.path.join(MODES_PATH,r"ShapeFile\YangtzeValley\YangtzeValley")
        m.readshapefile(fn1,'',drawbounds=True,linewidth=0.5,color='Green') #,color='gray'

        fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_BOUND")
        df.draw_map_lines(m,fn1,color='navy',linewidth=1.)

        fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_LVL1_RIVER_Clip")
        m.readshapefile(fn1,'',drawbounds=True,color='b',linewidth=0.8)

        fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_LVL2_RIVER")
        df.draw_map_lines(m,fn1,color='b',linewidth=.5)

        fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_LVL3_RIVER")
        df.draw_map_lines(m,fn1,color='b',linewidth=.3)

        #fn1 = os.path.join(MODES_PATH,r"ShapeFile\CJ_GIS_INFO\CJ_LVL4_RIVER")
        #df.draw_map_lines(m,fn1,color='b',linewidth=.2)

    #plt.title(u'填充示例图' ,fontproperties=cnfont)
    #######################################################
    from numpy.random import randn
    data = np.clip(randn(25, 25), -1, 1)
    #from matplotlib import cm
    #cax = ax.imshow(data, interpolation='nearest', colors=cmap2)
    cax = plt.contourf(data,colors=cmap2,levels=lev)
    #ax.set_title('Gaussian noise with vertical colorbar')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = plt.colorbar(cax, shrink=0.3)
    #######################################################
    cnfont = dclim.GetCnFont()
    plt.title(Title1.decode('gbk'),fontproperties=cnfont)
    imgfile='out.png'
    #DefaultSize = plt.get_size_inches()
    #print(DefaultSize)
    #plt.savefig(imgfile,dpi=300)
    plt.savefig(imgfile,dpi=200)
    df.CutPicWhiteBorder(imgfile)
    plt.show()


def GetAreaInfo(FileName):
    f=open(FileName)
    List1 = f.readlines()
    List2 =[]
    for txt in List1:
        #print(txt)
        txt.strip()
        l1 = txt.split()
        L2=[]
        #StaType.append(int(l1[1]))
        #AreaName.append(l1[2])
        #Value.append(l1[3])
        L2.append(int(l1[0]))
        L2.append(l1[1].decode('gb2312'))
        L2.append(string.atof(l1[2]))
        #print(type(string.atof(l1[2])))
        List2.append(L2)
    f.close

    print(List2)
    #sys.exit(0)
    return List2

def GetAreaRainColor(maplev,ValOut):
    cmap2 = maplev[:,:-1]
    cmap2 = cmap2/255.0
    lev = maplev[:,-1]
    ###print('lev=',lev)
    ###print('ValOut=',ValOut)

    p2 = np.where(lev<=ValOut,True,False)  #用于过滤的信度检验数组
    p3 = cmap2[p2]
    #########Debug Info#########
    #print(p2)
    #print(p3)
    #print('P3 = ',p3[-1])

    #sys.exit(0)
    from random import uniform
    #color =rgb2hex((uniform(0.5,1),uniform(0.5,1),uniform(0.5,1)))
    #print(p3[-1])
    color =rgb2hex(p3[-1])
    return(color)

def getiniLevel(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    Title = config1.get('Para',"Level")
    return Title

def getiniRainType(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    RainType = config1.get('Para',"RainType")
    return RainType

def getLevelFile(FileName,ParaStr):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    LevelFile = config1.get(ParaStr,"LevelFile")
    return(LevelFile)

def getXYArea(FileName,ParaStr):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    limitStr = config1.get(ParaStr,"DrawArea")
    ShpFile =  config1.get(ParaStr,"ShpFile")
    list1 = limitStr.split(',');
    #print(list1)
    return(float(list1[0]),float(list1[1]),float(list1[2]),float(list1[3]),float(list1[4]),ShpFile)


def getiniTitle(FileName):
    config1 = ConfigParser.ConfigParser()
    config1.readfp(open(FileName))
    Title = config1.get("Para","Title")
    return Title


def GetDict():
    dict1={}
    #Level=Level01
    #RainType=RainType01

    #MODES_PATH = os.environ.get('SKY_RAIN_MONI')
    MODES_PATH='.'
    MODES_INI_File = os.path.join( MODES_PATH,'RainMoni.ini')
    MODES_INI4Pred = os.path.join( MODES_PATH,'Moni4DrawMap.ini')
    print(MODES_INI_File)
    dict1['Level']=getiniLevel(MODES_INI4Pred)
    dict1['RainType']=getiniRainType(MODES_INI4Pred)
    dict1['LevelFile'] = getLevelFile(MODES_INI_File,dict1['RainType'])

    x1,x2,y1,y2,lon0,ShpFile = getXYArea(MODES_INI_File,dict1['Level'])
    dict1['xmin'],dict1['xmax']=x1,x2
    dict1['ymin'],dict1['ymax']=y1,y2
    dict1['lon0']=lon0
    dict1['ShpFile']=os.path.join(MODES_PATH,ShpFile)
    print(dict1)
    #sys.exit(0)

    return dict1

if __name__ == '__main__':
    Draw_Area_Rain()