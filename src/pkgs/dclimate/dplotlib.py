#-*-coding:utf8-*-
import sys
import numpy as np
import dfunc as df
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

'''
绘制高度场的函数库，作者杜良敏
编写时间：2010-2013
'''

def drawhigh(hgt,lons=np.arange(0, 360, 2.5, dtype=float),\
                 lats=np.arange(90, -90 - 1, -2.5, dtype=float),\
                 ptype=1,ptitle='',imgfile='tmp.png',showimg=0,cmap_str=None,lev=None,LatLonRange=[-90,90,0,360]):
    #cmap_str 颜色条
    #lev 为显示的级别
  lons,lats = np.meshgrid(lons, lats)
  fig=plt.figure(figsize=(12,6))
  # setup of sinusoidal basemap

  #m = Basemap(resolution='c',projection='sinu',lon_0=0)

  #LatLonRange

  if(ptype==1):
    m = Basemap(projection='cyl',llcrnrlat=LatLonRange[0],urcrnrlat=LatLonRange[1],\
            llcrnrlon=LatLonRange[2],urcrnrlon=LatLonRange[3]-2.5,resolution='c')
  #去处2.5避免白边

  if(ptype==2):
    m = Basemap(projection='npstere',boundinglat=0,lon_0=110,\
            resolution='c',area_thresh=10000.)
  if(ptype==3):
    m = Basemap(lon_0=-60,lat_0=90,projection='ortho',resolution='c')

  #ax = fig.add_axes([0.1,0.1,0.7,0.7])
  #ax = fig.add_axes([0.1,0.1,0.7,0.8])
  ax = fig.add_axes([0.1,0.1,0.7,0.8])
  # make a filled contour plot.

  x, y = m(lons, lats)

  #print 'x=',x
  #print 'y=',y

  print hgt.shape
  print x.shape,y.shape

  #CS = m.contour(x,y,hgt,15,linewidths=1) #,colors=plt.cm.jet) #'k')

  ##############################################################################
  if(lev==None):
    CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k')
  else:
    CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k',levels=lev)

  plt.clabel(CS,fmt='%6.0f')
  ##############################################################################

  #plt.clabel(CS, inline=2, fontsize=10)
  
  ##############################################################################
  if(cmap_str==None and lev==None):
    CS = m.contourf(x,y,hgt,15,cmap=plt.cm.jet,extend='both')

  if(cmap_str!=None and lev==None):
    CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),extend='both')

  if(cmap_str==None and lev!=None):
    CS = m.contourf(x,y,hgt,15,levels=lev,extend='both')

  if(cmap_str!=None and lev!=None):
    CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),levels=lev,extend='both')
  ##############################################################################
  
  m.drawcoastlines(linewidth=0.8)
  m.drawmapboundary()
  #m.fillcontinents()  #���ر���

  # setup colorbar axes instance.
  pos = ax.get_position()
  l, b, w, h = pos.bounds
  print 'b=',b
  print 'h=',h
  ColarbarAxes = plt.axes([l+w+0.01, 0.3, 0.02, 0.4]) # setup colorbar axes
  plt.colorbar(drawedges=True, cax=ColarbarAxes) # draw colorbar
  #plt.colorbar(orientation='vertical',shrink=0.6,ticks=lev2,drawedges=True,)
  plt.axes(ax)  # make the original axes current again
  # draw coastlines and political boundaries.
  # draw parallels and meridians.

  parallels = np.arange(-60.,90,30.)
  m.drawparallels(parallels,labels=[1,0,0,0])
  meridians = np.arange(-360.,360.,45.)
  m.drawmeridians(meridians,labels=[0,0,0,1])

  #parallels = np.arange(0.,80,20.)
  #m.drawparallels(parallels,labels=[0,0,1,1])
  #meridians = np.arange(10.,360.,20.)
  #m.drawmeridians(meridians,labels=[1,1,1,1])

  plt.title(ptitle,y=1.075, fontproperties='SimHei')
  print 'plotting with basemap ...'
  fig.savefig(imgfile, dpi=180)
  #fig.savefig(imgfile.replace('.png','.pdf'))
  #fig.savefig(imgfile.replace('.png','.svg'))
  #fig.savefig(imgfile.replace('.png','.eps'))
  df.CutPicWhiteBorder(imgfile)
  if(showimg):
    plt.show()
  plt.close()
#----------------------------------------------------------------------
def drawhigh4corr(hgt,lons,lats,ptype=1,ptitle='',imgfile='',showimg=0):
  lons,lats = np.meshgrid(lons, lats)
  fig=plt.figure(figsize=(12,6))
  # setup of sinusoidal basemap

  #m = Basemap(resolution='c',projection='sinu',lon_0=0)

  if(ptype==1):
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=0,urcrnrlon=360-2.5,resolution='c')
  if(ptype==2):
    m = Basemap(projection='npstere',boundinglat=0,lon_0=90,\
            resolution='c',area_thresh=10000.)
  if(ptype==3):
    m = Basemap(lon_0=-60,lat_0=90,projection='ortho')

  #ax = fig.add_axes([0.1,0.1,0.7,0.7])
  #ax = fig.add_axes([0.1,0.1,0.7,0.8])
  ax = fig.add_axes([0.1,0.1,0.7,0.8])
  # make a filled contour plot.

  x, y = m(lons, lats)

  #print 'x=',x
  #print 'y=',y

  print hgt.shape
  print x.shape,y.shape
  #lev =[-1, -0.8, -0.5, -0.2, 0 ,0.2, 0.5, 0.8, 1]
  lev = np.linspace(-1,1,11)
  lev2= np.linspace(-0.9,0.9,21-2)
  #CS = m.contour(x,y,hgt,15,linewidths=0.5) #,colors=plt.cm.jet) #'k')

  CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k',levels=lev2)
  #plt.clabel(CS,fmt='%6.0f')

  plt.clabel(CS, fmt='%4.2f',inline=2, fontsize=8)

  CS = m.contourf(x,y,hgt,15,cmap=plt.cm.bwr,extend='both',levels=lev)

  m.drawcoastlines()
  m.drawmapboundary()
  #m.fillcontinents()  #���ر���

  # setup colorbar axes instance.
  pos = ax.get_position()
  l, b, w, h = pos.bounds
  print 'b=',b
  print 'h=',h
  #ColarbarAxes = plt.axes([l+w+0.01, 0.3, 0.02, 0.4]) # setup colorbar axes

  plt.colorbar(drawedges=True,shrink=0.4, ticks=np.array(lev)) # draw colorbar  cax=ColarbarAxes,
  #plt.colorbar(orientation='vertical',shrink=0.6,ticks=lev2,drawedges=True,)
  plt.axes(ax)  # make the original axes current again
  # draw coastlines and political boundaries.
  # draw parallels and meridians.

  parallels = np.arange(-60.,90,30.)
  m.drawparallels(parallels,labels=[1,0,0,0])
  meridians = np.arange(-360.,360.,45.)
  m.drawmeridians(meridians,labels=[0,0,0,1])

  #parallels = np.arange(0.,80,20.)
  #m.drawparallels(parallels,labels=[0,0,1,1])
  #meridians = np.arange(10.,360.,20.)
  #m.drawmeridians(meridians,labels=[1,1,1,1])

  import dclimate as dclim
  cnfont = dclim.GetCnFont()

  plt.title(ptitle,y=1.075, fontproperties=cnfont)
  print 'plotting with basemap ...'
  if(imgfile!=''):
    if(os.path.exists(r'c:\convert.exe')):
        fig.savefig('tmp1.png', dpi=160)
        #切除白边
        str1 = 'c:\convert.exe tmp1.png -trim +repage %s'%(imgfile)
        os.system(str1)
    else:
        fig.savefig(imgfile, dpi=160)

  df.CutPicWhiteBorder(imgfile)
  if(showimg):
    plt.show()

#----------------------------------------------------------------------

#self.lons
#self.lats =

def drawhigh4corr2(rval,pval,**args): #lons,lats,ptype=1,ptitle='',imgfile='',showimg=0):
    """

    """
    lons = args.pop("lons",np.arange(0, 360, 2.5, dtype=float))
    lats = args.pop("lats",np.arange(90, -90 - 1, -2.5, dtype=float))

    ptype = args.pop("ptype",1)
    ptitle = args.pop("ptitle",'')
    imgfile = args.pop("imgfile",'')
    showimg = args.pop("showimg",0)

    lons,lats = np.meshgrid(lons, lats)
    fig=plt.figure(figsize=(12,6))
    # setup of sinusoidal basemap

    #m = Basemap(resolution='c',projection='sinu',lon_0=0)
    if(ptype==1):
        m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=0,urcrnrlon=360-2.5,resolution='c')
    if(ptype==2):
        m = Basemap(projection='npstere',boundinglat=0,lon_0=90,\
                resolution='c',area_thresh=10000.)
    if(ptype==3):
        m = Basemap(lon_0=-60,lat_0=90,projection='ortho')

    #ax = fig.add_axes([0.1,0.1,0.7,0.7])
    #ax = fig.add_axes([0.1,0.1,0.7,0.8])
    ax = fig.add_axes([0.1,0.1,0.7,0.8])
    # make a filled contour plot.

    x, y = m(lons, lats)

    #print 'x=',x
    #print 'y=',y

    print rval.shape
    print x.shape,y.shape
    #lev =[-1, -0.8, -0.5, -0.2, 0 ,0.2, 0.5, 0.8, 1]
    lev = np.linspace(-1,1,11) #间距
    levf = np.linspace(-1,1,41)
    lev2= np.linspace(-0.9,0.9,21-2)
    #CS = m.contour(x,y,hgt,15,linewidths=0.5) #,colors=plt.cm.jet) #'k')

    CS = m.contour(x,y,pval,15,linewidths=0.6,colors='k',levels= np.array([0.05,0.01]))
    #plt.clabel(CS,fmt='%6.0f')

    plt.clabel(CS, fmt='%4.2f',inline=2, fontsize=8)

    CS = m.contourf(x,y,rval,15,cmap=plt.cm.bwr,extend='both',levels=levf)

    m.drawcoastlines()
    m.drawmapboundary()
    #m.fillcontinents()  #���ر���

    # setup colorbar axes instance.
    pos = ax.get_position()
    l, b, w, h = pos.bounds
    print 'b=',b
    print 'h=',h
    #ColarbarAxes = plt.axes([l+w+0.01, 0.3, 0.02, 0.4]) # setup colorbar axes

    plt.colorbar(drawedges=True,shrink=0.4, ticks=np.array(lev)) # draw colorbar  cax=ColarbarAxes,
    #plt.colorbar(orientation='vertical',shrink=0.6,ticks=lev2,drawedges=True,)
    plt.axes(ax)  # make the original axes current again
    # draw coastlines and political boundaries.
    # draw parallels and meridians.

    parallels = np.arange(-60.,90,30.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-360.,360.,45.)
    m.drawmeridians(meridians,labels=[0,0,0,1])

    #parallels = np.arange(0.,80,20.)
    #m.drawparallels(parallels,labels=[0,0,1,1])
    #meridians = np.arange(10.,360.,20.)
    #m.drawmeridians(meridians,labels=[1,1,1,1])

    plt.title(ptitle,y=1.075, fontproperties='SimHei')
    print 'plotting with basemap ...'
    if(imgfile!=''):
        fig.savefig(imgfile, dpi=160)

    df.CutPicWhiteBorder(imgfile)
    if(showimg):
        plt.show()

def drawhigh5880Line(hgt,hgt_avg,lons,lats,ptype=1,ptitle='',imgfile='',showimg=0,cmap_str=None,lev=None):
    #cmap_str 颜色条
    #lev 为显示的级别
  lons,lats = np.meshgrid(lons, lats)
  fig=plt.figure(figsize=(12,6))
  # setup of sinusoidal basemap

  #m = Basemap(resolution='c',projection='sinu',lon_0=0)

  if(ptype==1):
    #m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=60,\
    m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=50,\
            llcrnrlon=90,urcrnrlon=180,resolution='c')
  #去处2.5避免白边

  if(ptype==2):
    m = Basemap(projection='npstere',boundinglat=0,lon_0=90,\
            resolution='c',area_thresh=10000.)
  if(ptype==3):
    m = Basemap(lon_0=-60,lat_0=90,projection='ortho')

  #ax = fig.add_axes([0.1,0.1,0.7,0.7])
  #ax = fig.add_axes([0.1,0.1,0.7,0.8])
  ax = fig.add_axes([0.1,0.1,0.7,0.8])
  # make a filled contour plot.

  x, y = m(lons, lats)

  #print 'x=',x
  #print 'y=',y

  print hgt.shape
  print x.shape,y.shape

  #CS = m.contour(x,y,hgt,15,linewidths=1) #,colors=plt.cm.jet) #'k')

  ##############################################################################
  #CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k')
  CS2 = m.contour(x,y,hgt_avg,15,linewidths=5,levels=lev,linestyles='dashed',color='b') #cmap=plt.cm.bwr

  #cs = pl.contour(x1, x2, PHI(- y_pred / sigma), [0.5], colors='k', \
  #              linestyles='dashed')

  #plt.clabel(CS2,fmt='%6.0f',inline=1)
  clbls = plt.clabel(CS2, fmt="%6.0f", use_clabeltext=True)
  #非常重要
  import matplotlib.patheffects as PathEffects

  plt.setp(clbls,
             path_effects=[PathEffects.withStroke(linewidth=1,
                                                  foreground="w")])



  CS = m.contour(x,y,hgt,15,linewidths=5,levels=lev,colors='r')
  plt.clabel(CS,fmt='%6.0f',inline=1)



  ##############################################################################

  #plt.clabel(CS, inline=2, fontsize=10)

  ##############################################################################
  if(0):
    if(cmap_str==None and lev==None):
        CS = m.contourf(x,y,hgt,15,cmap=plt.cm.jet,extend='both')
    if(cmap_str!=None and lev==None):
        CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),extend='both')
    if(cmap_str==None and lev!=None):
        CS = m.contourf(x,y,hgt,15,levels=lev,extend='both')
    if(cmap_str!=None and lev!=None):
        CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),levels=lev,extend='both')
  ##############################################################################

  m.drawcoastlines()
  m.drawmapboundary()
  #m.fillcontinents()  #���ر���

  # setup colorbar axes instance.
  pos = ax.get_position()
  l, b, w, h = pos.bounds
  print 'b=',b
  print 'h=',h
  #ColarbarAxes = plt.axes([l+w+0.01, 0.3, 0.02, 0.4]) # setup colorbar axes
  #plt.colorbar(drawedges=True, cax=ColarbarAxes) # draw colorbar
  #plt.colorbar(orientation='vertical',shrink=0.6,ticks=lev2,drawedges=True,)
  #plt.axes(ax)  # make the original axes current again
  # draw coastlines and political boundaries.
  # draw parallels and meridians.

  parallels = np.arange(-60.,90,15.)
  m.drawparallels(parallels,labels=[1,0,0,0])
  meridians = np.arange(-360.,360.,45.)
  m.drawmeridians(meridians,labels=[0,0,0,1])

  #parallels = np.arange(0.,80,20.)
  #m.drawparallels(parallels,labels=[0,0,1,1])
  #meridians = np.arange(10.,360.,20.)
  #m.drawmeridians(meridians,labels=[1,1,1,1])

  plt.title(ptitle,y=1.075, fontproperties='SimHei')
  print 'plotting with basemap ...'
  if(imgfile!=''):
    #fig.savefig(imgfile, dpi=120)
    if(os.path.exists(r'c:\convert.exe')):
        fig.savefig('tmp1.png', dpi=160)
        #切除白边
        str1 = 'c:\convert.exe tmp1.png -trim +repage %s'%(imgfile)
        os.system(str1)
    else:
        fig.savefig(imgfile, dpi=160)
  df.CutPicWhiteBorder(imgfile)
  if(showimg):
    plt.show()

#----------------------------------------------------------------------
def drawhigh_split(hgt,lons=np.arange(0, 360, 2.5, dtype=float), \
             lats=np.arange(90, -90 - 1, -2.5, dtype=float), \
             ptype=1,ptitle='',imgfile='',showimg=0,cmap_str=None,lev=None):
#cmap_str 颜色条
#lev 为显示的级别
    lons,lats = np.meshgrid(lons, lats)
    fig=plt.figure(figsize=(12,6))
    # setup of sinusoidal basemap

    #m = Basemap(resolution='c',projection='sinu',lon_0=0)

    if(ptype==1):
        m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90, \
                    llcrnrlon=0,urcrnrlon=360-2.5,resolution='c')
        #去处2.5避免白边

    if(ptype==2):
        m = Basemap(projection='npstere',boundinglat=0,lon_0=90, \
                    resolution='c',area_thresh=10000.)
    if(ptype==3):
        m = Basemap(lon_0=-60,lat_0=90,projection='ortho')

    #ax = fig.add_axes([0.1,0.1,0.7,0.7])
    #ax = fig.add_axes([0.1,0.1,0.7,0.8])
    ax = fig.add_axes([0.1,0.1,0.7,0.8])
    # make a filled contour plot.

    x, y = m(lons, lats)

    #print 'x=',x
    #print 'y=',y

    print hgt.shape
    print x.shape,y.shape

    #CS = m.contour(x,y,hgt,15,linewidths=1) #,colors=plt.cm.jet) #'k')

    ##############################################################################
    if(lev==None):
        #CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k')
        pass
    else:
        CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k',levels=lev)
        plt.clabel(CS,fmt='%6.0f')

    ##############################################################################

    #plt.clabel(CS, inline=2, fontsize=10)

    ##############################################################################
    if(cmap_str==None and lev==None):
        #CS = m.contourf(x,y,hgt,15,cmap=plt.cm.jet,extend='both')
        cluster_num = np.size(np.unique(hgt))
        #cmap = plt.get_cmap('OrRd',cluster_num)
        cmap = plt.get_cmap('jet',cluster_num)
        cmap.set_bad(color='w',alpha=1)
        hgt1 = hgt.copy()
        hgt = np.where(hgt==0,np.nan,hgt)
        #np.savetxt('c2.txt',hgt)

        #iml = m.pcolor(x,y,hgt,cmap=cmap,levels=np.unique(np.ravel(hgt1))) # norm=plt.Normalize(1,cluster_num))
        iml = m.contourf(x,y,hgt,cmap=cmap,levels=np.unique(np.ravel(hgt1))) # norm=plt.Normalize(1,cluster_num))
        #iml = m.pcolormesh(x,y,hgt,cmap=cmap, norm=plt.Normalize(1,cluster_num))
        #iml = m.pcolor(x,y,hgt,cmap=cmap, norm=plt.Normalize(1,cluster_num))
        #iml = m.imshow(x,y,hgt)

    if(cmap_str!=None and lev==None):
        CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),extend='both')

    if(cmap_str==None and lev!=None):
        cluster_num = np.size(np.unique(hgt))
        #cmap = plt.get_cmap('OrRd',cluster_num)
        cmap = plt.get_cmap('hsv',cluster_num) #autumn
        cmap.set_bad(color='w',alpha=1)
        hgt1 = hgt.copy()
        hgt = np.where(hgt==0,np.nan,hgt)
        #iml = m.pcolormesh(x,y,hgt,cmap=cmap, norm=plt.Normalize(1,np.size(lev)))
        iml = m.contourf(x,y,hgt,cmap=cmap,levels=lev) # norm=plt.Normalize(1,cluster_num))
        #np.savetxt('c3.txt',hgt)

        #iml = m.contourf(x,y,hgt,cmap=cmap,lev=np.unique(np.ravel(hgt1))) # norm=plt.Normalize(1,cluster_num))

    if(cmap_str!=None and lev!=None):
        CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),levels=lev,extend='both')
        ##############################################################################

    m.drawcoastlines()
    m.drawmapboundary()
    #m.fillcontinents()  #���ر���

    # setup colorbar axes instance.
    pos = ax.get_position()
    l, b, w, h = pos.bounds
    print 'b=',b
    print 'h=',h
    ColarbarAxes = plt.axes([l+w+0.01, 0.3, 0.02, 0.4]) # setup colorbar axes
    plt.colorbar(drawedges=True, cax=ColarbarAxes) # draw colorbar
    #plt.colorbar(orientation='vertical',shrink=0.6,ticks=lev2,drawedges=True,)
    plt.axes(ax)  # make the original axes current again
    # draw coastlines and political boundaries.
    # draw parallels and meridians.

    parallels = np.arange(-60.,90,30.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-360.,360.,45.)
    m.drawmeridians(meridians,labels=[0,0,0,1])

    #parallels = np.arange(0.,80,20.)
    #m.drawparallels(parallels,labels=[0,0,1,1])
    #meridians = np.arange(10.,360.,20.)
    #m.drawmeridians(meridians,labels=[1,1,1,1])

    plt.title(ptitle,y=1.075, fontproperties='SimHei')
    print 'plotting with drawhigh_split basemap ...'
    fig.savefig(imgfile, dpi=180)
    df.CutPicWhiteBorder(imgfile)
    if(showimg):
        plt.show()
    plt.close()
    #----------------------------------------------------------------------



#----------------------------------------------------------------------
def drawhigh_split2(hgt,position,lons=np.arange(0, 360, 2.5, dtype=float), \
             lats=np.arange(90, -90 - 1, -2.5, dtype=float), \
             ptype=1,ptitle='',imgfile='',showimg=0,cmap_str=None,lev=None):
#cmap_str 颜色条
#lev 为显示的级别
    lons,lats = np.meshgrid(lons, lats)
    fig=plt.figure(figsize=(12,6))
    # setup of sinusoidal basemap

    #m = Basemap(resolution='c',projection='sinu',lon_0=0)

    if(ptype==1):
        m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90, \
                    llcrnrlon=0,urcrnrlon=360-2.5,resolution='c')
        #去处2.5避免白边

    if(ptype==2):
        m = Basemap(projection='npstere',boundinglat=0,lon_0=90, \
                    resolution='c',area_thresh=10000.)
    if(ptype==3):
        m = Basemap(lon_0=-60,lat_0=90,projection='ortho')

    #ax = fig.add_axes([0.1,0.1,0.7,0.7])
    #ax = fig.add_axes([0.1,0.1,0.7,0.8])
    ax = fig.add_axes([0.1,0.1,0.7,0.8])
    # make a filled contour plot.

    x, y = m(lons, lats)

    #print 'x=',x
    #print 'y=',y


    print hgt.shape
    print x.shape,y.shape


    #CS = m.contour(x,y,hgt,15,linewidths=1) #,colors=plt.cm.jet) #'k')

    ##############################################################################
    if(lev==None):
        #CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k')
        pass
    else:
        CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k',levels=lev)
        plt.clabel(CS,fmt='%6.0f')

    ##############################################################################

    #plt.clabel(CS, inline=2, fontsize=10)

    ##############################################################################
    if(cmap_str==None and lev==None):
        #CS = m.contourf(x,y,hgt,15,cmap=plt.cm.jet,extend='both')
        cluster_num = np.size(np.unique(hgt))
        #cmap = plt.get_cmap('OrRd',cluster_num)
        cmap = plt.get_cmap('jet',cluster_num)
        cmap.set_bad(color='w',alpha=1)
        hgt1 = hgt.copy()
        hgt = np.where(hgt==0,np.nan,hgt)
        #np.savetxt('c2.txt',hgt)

        #iml = m.pcolor(x,y,hgt,cmap=cmap,levels=np.unique(np.ravel(hgt1))) # norm=plt.Normalize(1,cluster_num))
        iml = m.contourf(x,y,hgt,cmap=cmap,levels=np.unique(np.ravel(hgt1))) # norm=plt.Normalize(1,cluster_num))
        #iml = m.pcolormesh(x,y,hgt,cmap=cmap, norm=plt.Normalize(1,cluster_num))
        #iml = m.pcolor(x,y,hgt,cmap=cmap, norm=plt.Normalize(1,cluster_num))
        #iml = m.imshow(x,y,hgt)

    if(cmap_str!=None and lev==None):
        CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),extend='both')

    if(cmap_str==None and lev!=None):
        cluster_num = np.size(np.unique(hgt))
        #cmap = plt.get_cmap('OrRd',cluster_num)
        cmap = plt.get_cmap('hsv',cluster_num) #autumn
        cmap.set_bad(color='w',alpha=1)
        hgt1 = hgt.copy()
        hgt = np.where(hgt==0,np.nan,hgt)
        #iml = m.pcolormesh(x,y,hgt,cmap=cmap, norm=plt.Normalize(1,np.size(lev)))
        iml = m.contourf(x,y,hgt,cmap=cmap,levels=lev) # norm=plt.Normalize(1,cluster_num))
        #np.savetxt('c3.txt',hgt)

        #iml = m.contourf(x,y,hgt,cmap=cmap,lev=np.unique(np.ravel(hgt1))) # norm=plt.Normalize(1,cluster_num))

    if(cmap_str!=None and lev!=None):
        CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),levels=lev,extend='both')
        ##############################################################################
    #plt.text(x[0,0],y[0,0],'AAAA')

    #print(position['lat'])
    #print(position['lon'])
    #x1,y1 = m(position['lon'],position['lat'])
    #plt.text(x1,y1,'AAA')

    for ii in range(position.shape[0]):
        print(position.iloc[ii])
        print(lons)
        print(lats)
        x1,y1 = m(position.iloc[ii]['lon'],to_lat1( position.iloc[ii]['lat']-5) )

        plt.text(x1,y1,'%d'%(ii+1),fontsize=16,color='b',fontweight='bold',va='top')


    #sys.exit(0)

    m.drawcoastlines()
    m.drawmapboundary()

    #m.fillcontinents()  #���ر���

    # setup colorbar axes instance.
    pos = ax.get_position()
    l, b, w, h = pos.bounds
    print 'b=',b
    print 'h=',h
    ColarbarAxes = plt.axes([l+w+0.01, 0.3, 0.02, 0.4]) # setup colorbar axes
    plt.colorbar(drawedges=True, cax=ColarbarAxes) # draw colorbar
    #plt.colorbar(orientation='vertical',shrink=0.6,ticks=lev2,drawedges=True,)
    plt.axes(ax)  # make the original axes current again
    # draw coastlines and political boundaries.
    # draw parallels and meridians.

    parallels = np.arange(-60.,90,30.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-360.,360.,45.)
    m.drawmeridians(meridians,labels=[0,0,0,1])

    #parallels = np.arange(0.,80,20.)
    #m.drawparallels(parallels,labels=[0,0,1,1])
    #meridians = np.arange(10.,360.,20.)
    #m.drawmeridians(meridians,labels=[1,1,1,1])

    plt.title(ptitle,y=1.075, fontproperties='SimHei')
    print 'plotting with drawhigh_split basemap ...'
    fig.savefig(imgfile, dpi=180)
    #fig.savefig(imgfile.replace('.png','.pdf'))
    df.CutPicWhiteBorder(imgfile)
    if(showimg):
        plt.show()
    plt.close()
    #sys.exit(0)
    #----------------------------------------------------------------------

def to_lat1(a1):
    if(a1>90):
        b1 = -1*(a1-90)
        return b1
    else:
        return a1

#----------------------------------------------------------------------
def draw_grb_china(lats,lons,data,title1,imgfile,levfile='1.LEV'):
    #lats,lons = grb.latlons()
    #data = grb.values
    from __init__ import Spatial_Data,Station_Data,Level_Path
    import dgriddata as dgrid
    import dfunc as dfunc
    print(data.min(), data.max())
    #m = Basemap(lon_0=180,projection='cyl')
    m = Basemap(llcrnrlon=86,llcrnrlat=13.5,urcrnrlon=146.5,urcrnrlat=51, \
        projection='lcc',lon_0=110,lat_0=30.0,lat_1=60.0,resolution='l') #resolution=None

    m.drawparallels(np.arange(20,71,10),labels=[1,0,0,0],linewidth=0.2, fontsize=8)
    #m.drawparallels(circles,labels=[1,0,0,0])
    m.drawmeridians(np.arange(80,131,10),labels=[0,1,0,1,0,1],linewidth=0.2, fontsize=8)
    #m.drawmeridians(meridians,labels=[0,0,0,1])
    m.drawcoastlines(linewidth=0.2)

    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"china_province"))

    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"huanghe_ganliu"),color='b',linewidth=1.0)

    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_LVL1_RIVER"),color='b',linewidth=1.0)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_BOUND"),color='g',linewidth=0.5)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"china_border\hubei"),color='r',linewidth=0.5)

    xi, yi = m(lons, lats)
    #CS = m.contourf(x,y,data,15,cmap=plt.cm.jet)

    print('levfile=',levfile)
    if( os.path.isfile(levfile)):
        #maplev = np.loadtxt('maplev_RAP.lev')
        maplev = np.loadtxt(levfile)
        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)
        #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        cax = m.contourf(xi,yi,data,colors=cmap2,levels=lev,extend='both')#plt.cm.jet, #max
        CS = m.contour(xi,yi,data,levels=lev,linewidths=0.5,colors='k')
        plt.colorbar(cax,shrink=0.6)
        plt.clabel(CS, fmt='%4.1f',inline=1,fontsize=8)
    else:
        CS = m.contourf(xi,yi,data,15,cmap=plt.cm.jet)
        plt.colorbar(CS,shrink=0.6)
        #plt.clabel(CS, fmt='%4.1f',inline=1,fontsize=8)

    m.drawmapboundary(fill_color='w',linewidth='0.5')
    #m.colorbar()
    m.drawcoastlines(linewidth='0.5')
    # draw parallels
    delat = 30.
    circles = np.arange(-90.,90.+delat,delat)

    # draw meridians
    delon = 60.
    meridians = np.arange(0,360,delon)

    plt.title(title1,size=8)
    #plt.show()
    plt.savefig(imgfile,dpi=180)
    dfunc.CutPicWhiteBorder(imgfile)
    #sys.exit(0)

    plt.close()


#----------------------------------------------------------------------
def draw_grb_world(lats,lons,data,title1,imgfile,levfile='1.LEV'):
    #lats,lons = grb.latlons()
    #data = grb.values
    from __init__ import Spatial_Data,Station_Data,Level_Path
    import dgriddata as dgrid
    import dfunc as dfunc
    print(data.min(), data.max())
    #m = Basemap(lon_0=180,projection='cyl')
    #m = Basemap(llcrnrlon=86,llcrnrlat=13.5,urcrnrlon=146.5,urcrnrlat=51, \
    #    projection='lcc',lon_0=110,lat_0=30.0,lat_1=60.0,resolution='l') #resolution=None

    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')

    #m.drawparallels(np.arange(20,71,10),labels=[1,0,0,0],linewidth=0.2, fontsize=8)
    #m.drawmeridians(np.arange(80,131,10),labels=[0,1,0,1,0,1],linewidth=0.2, fontsize=8)


    parallels = np.arange(-60.,90,30.)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = np.arange(-360.,360.,45.)
    m.drawmeridians(meridians,labels=[0,0,0,1])


    m.drawcoastlines(linewidth=0.2)

    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"china_province"))

    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"huanghe_ganliu"),color='b',linewidth=1.0)

    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_LVL1_RIVER"),color='b',linewidth=1.0)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_BOUND"),color='g',linewidth=0.5)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"china_border\hubei"),color='r',linewidth=0.5)

    xi, yi = m(lons, lats)
    #CS = m.contourf(x,y,data,15,cmap=plt.cm.jet)

    print('levfile=',levfile)
    if( os.path.isfile(levfile)):
        #maplev = np.loadtxt('maplev_RAP.lev')
        maplev = np.loadtxt(levfile)
        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)
        #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        cax = m.contourf(xi,yi,data,colors=cmap2,levels=lev,extend='both')#plt.cm.jet, #max
        CS = m.contour(xi,yi,data,levels=lev,linewidths=0.5,colors='k')
        plt.colorbar(cax,shrink=0.6)
        plt.clabel(CS, fmt='%4.1f',inline=1,fontsize=8)
    else:
        CS = m.contourf(xi,yi,data,15,cmap=plt.cm.jet)
        plt.colorbar(CS,shrink=0.6)
        #plt.clabel(CS, fmt='%4.1f',inline=1,fontsize=8)

    m.drawmapboundary(fill_color='w',linewidth='0.5')
    #m.colorbar()
    m.drawcoastlines(linewidth='0.5')
    # draw parallels
    delat = 30.
    circles = np.arange(-90.,90.+delat,delat)

    # draw meridians
    delon = 60.
    meridians = np.arange(0,360,delon)
    '''
    #寒潮关键区
    x4 = np.array([70.,70,90,90.,70]) #经度
    y4 = np.array([43.,65,65,43.,43]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color=np.array([255,127,127])/255.0,linewidth='1')
    print('m.plot(x4')


    #乌山阻高
    x4 = np.array([40.,40,70,70.,40]) #经度
    y4 = np.array([40.,50,50,40.,40]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')
    #贝湖阻高
    x4 = np.array([80.,80,110,110.,80]) #经度
    y4 = np.array([50.,60,60,50.,50]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')

    #鄂海阻高
    x4 = np.array([120.,120,150,150.,120]) #经度
    y4 = np.array([50.,60,60,50.,50]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')

    #西伯利亚高压
    x4 = np.array([80.,80,120,120.,80]) #经度
    y4 = np.array([40.,60,60,40.,40]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')
    '''
    #冷空气关键区
    x4 = np.array([45.,45,90,90.,45]) #经度
    y4 = np.array([60.,70,70,60.,60]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='r',linewidth='1')
    print('m.plot(x4')

    #青藏高原指数
    x5 = np.array([80.,80,110,110,80]) #经度
    y5 = np.array([30.,45,45,30,30])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='r',linewidth='1')

    #印缅槽
    #(15o-20oN,80o-100oE
    x5 = np.array([80.,80,100,100,80]) #经度
    y5 = np.array([15.,20,20,15,15])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth=1)

    #马斯克林高压
    #(15o-20oN,80o-100oE
    x5 = np.array([46.,46,86,86,46]) #经度
    y5 = np.array([-34,-26,-26,-34,-34])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth=1)


    #澳大利亚高压
    #(15o-20oN,80o-100oE
    x5 = np.array([124.,124,156,156,124]) #经度
    y5 = np.array([-36,-24,-24,-36,-36])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth=1)


    plt.title(title1,size=8)
    #plt.show()
    plt.savefig(imgfile,dpi=280)
    dfunc.CutPicWhiteBorder(imgfile)
    #sys.exit(0)

    plt.close()


def drawhigh_EA(hgt,lons=np.arange(0, 360, 2.5, dtype=float),\
                 lats=np.arange(90, -90 - 1, -2.5, dtype=float),\
                 ptype=1,ptitle='',imgfile='',showimg=0,cmap_str=None,levfile='',LatLonRange=[-90,90,0,360]):
    #cmap_str 颜色条
    #lev 为显示的级别
    from __init__ import Spatial_Data,Station_Data,Level_Path
    import dgriddata as dgrid
    lons,lats = np.meshgrid(lons, lats)
    fig=plt.figure(figsize=(12,6))
      # setup of sinusoidal basemap

    #m = Basemap(resolution='c',projection='sinu',lon_0=0)

    #LatLonRange

    if(ptype==1):
        m = Basemap(projection='cyl',llcrnrlat=LatLonRange[0],urcrnrlat=LatLonRange[1],\
            llcrnrlon=LatLonRange[2],urcrnrlon=LatLonRange[3]-2.5,resolution='c')
    #去处2.5避免白边

    if(ptype==2):
        m = Basemap(projection='npstere',boundinglat=0,lon_0=110,\
            resolution='c',area_thresh=10000.)

    if(ptype==3):
        m = Basemap(lon_0=-60,lat_0=90,projection='ortho',resolution='c')

    #ax = fig.add_axes([0.1,0.1,0.7,0.7])
    #ax = fig.add_axes([0.1,0.1,0.7,0.8])
    ax = fig.add_axes([0.1,0.1,0.7,0.8])
    # make a filled contour plot.

    x, y = m(lons, lats)

    #print 'x=',x
    #print 'y=',y

    print hgt.shape
    print x.shape,y.shape


    print('levfile=',levfile)
    if( os.path.isfile(levfile)):
        #maplev = np.loadtxt('maplev_RAP.lev')
        maplev = np.genfromtxt(levfile)
        cmap2 = maplev[:,:-1]
        cmap2 = cmap2/255.0
        lev = maplev[:,-1]
        #print(cmap2)
        #cax = m.contourf(xi,yi,zi,colors=cmap2,levels=lev,extend='both')#plt.cm.jet,
        cax = m.contourf(x,y,hgt,colors=cmap2,levels=lev,extend='both')#plt.cm.jet, #max
        CS = m.contour(x,y,hgt,levels=lev,linewidths=0.5,colors='k')
        plt.colorbar(cax,shrink=0.6)
        plt.clabel(CS, fmt='%4.1f',inline=1,fontsize=8)
    else:
        CS = m.contourf(x,y,hgt,15,cmap=plt.cm.jet)
        plt.colorbar(CS,shrink=0.6)
        #plt.clabel(CS, fmt='%4.1f',inline=1,fontsize=8)

    #CS = m.contour(x,y,hgt,15,linewidths=1) #,colors=plt.cm.jet) #'k')

    # ##############################################################################
    # if(lev==None):
    #   CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k')
    # else:
    #   CS = m.contour(x,y,hgt,15,linewidths=0.5,colors='k',levels=lev)
    #
    # plt.clabel(CS,fmt='%4.1f')
    # ##############################################################################
    #
    # #plt.clabel(CS, inline=2, fontsize=10)
    #
    # ##############################################################################
    # if(cmap_str==None and lev==None):
    #   CS = m.contourf(x,y,hgt,15,cmap=plt.cm.jet,extend='both')
    #
    # if(cmap_str!=None and lev==None):
    #   CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),extend='both')
    #
    # if(cmap_str==None and lev!=None):
    #       CS = m.contourf(x,y,hgt,15,levels=lev,extend='both')
    #
    # if(cmap_str!=None and lev!=None):
    #   CS = m.contourf(x,y,hgt,15,cmap=plt.get_cmap(cmap_str),levels=lev,extend='both')
    # ##############################################################################
    #plt.clabel(CS,fmt='%4.1f')

    m.drawcoastlines(linewidth=0.8)
    m.drawmapboundary()
    #m.fillcontinents()  #���ر���

    # setup colorbar axes instance.
    pos = ax.get_position()
    l, b, w, h = pos.bounds
    print 'b=',b
    print 'h=',h
    #ColarbarAxes = plt.axes([l+w+0.01, 0.3, 0.02, 0.4]) # setup colorbar axes
    #plt.colorbar(drawedges=True, cax=ColarbarAxes) # draw colorbar
    #plt.colorbar(orientation='vertical',shrink=0.6,ticks=lev2,drawedges=True,)
    plt.axes(ax)  # make the original axes current again
    # draw coastlines and political boundaries.
    # draw parallels and meridians.

    parallels = np.arange(-90.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.4,fontsize=5)
    meridians = np.arange(-360.,360.,10.)
    m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.4,fontsize=5)


    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"china_province"))
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"huanghe_ganliu"),color='b',linewidth=1.0)
    dgrid.draw_map_lines(m,os.path.join(Spatial_Data,"CJ_LVL1_RIVER"),color='b',linewidth=1.0)


    # x4 = np.array([40.,40,70,70.,40]) #纬度
    # y4 = np.array([50.,70,70,50.,50]) #经度
    # x4, y4 = m(x4,y4)
    # m.plot(x4,y4,color='g',linewidth='3')
    # print('m.plot(x4')
    '''
    #寒潮关键区
    x4 = np.array([70.,70,90,90.,70]) #经度
    y4 = np.array([43.,65,65,43.,43]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color=np.array([255,127,127])/255.0,linewidth='1')
    print('m.plot(x4')

    #乌山阻高
    x4 = np.array([40.,40,70,70.,40]) #经度
    y4 = np.array([40.,50,50,40.,40]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')
    #贝湖阻高
    x4 = np.array([80.,80,110,110.,80]) #经度
    y4 = np.array([50.,60,60,50.,50]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')

    #鄂海阻高
    x4 = np.array([120.,120,150,150.,120]) #经度
    y4 = np.array([50.,60,60,50.,50]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')

    #西伯利亚高压
    x4 = np.array([80.,80,120,120.,80]) #经度
    y4 = np.array([40.,60,60,40.,40]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='c',linewidth='1')
    print('m.plot(x4')

    #冷空气关键区 1
    x4 = np.array([45.,45,90,90.,45]) #经度
    y4 = np.array([60.,70,70,60.,60]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='g',linewidth='2')
    print('m.plot(x4')
    '''

    #冷空气关键区 2
    x4 = np.array([40.,40,85,85.,40]) #经度
    y4 = np.array([57.,73,73,57.,57]) #纬度
    x4, y4 = m(x4,y4)
    m.plot(x4,y4,color='g',linewidth='2')
    print('m.plot(x4')

    '''
    #青藏高原指数 1
    x5 = np.array([80.,80,110,110,80]) #经度
    y5 = np.array([30.,45,45,30,30])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth='2')

    #青藏高原指数 2
    x5 = np.array([75.,75,105,105,75]) #经度
    y5 = np.array([35.,50,50,35,35])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='g',linewidth='2')
    '''

    #青藏高原指数 3
    x5 = np.array([75.,75,105,105,75]) #经度
    y5 = np.array([33,48,48,33,33])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth='2')

    '''
    #印缅槽
    #(15o-20oN,80o-100oE
    x5 = np.array([80.,80,100,100,80]) #经度
    y5 = np.array([15.,20,20,15,15])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth=1)

    #马斯克林高压
    #(15o-20oN,80o-100oE
    x5 = np.array([46.,46,86,86,46]) #经度
    y5 = np.array([-34,-26,-26,-34,-34])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth=1)


    #澳大利亚高压
    #(15o-20oN,80o-100oE
    x5 = np.array([124.,124,156,156,124]) #经度
    y5 = np.array([-36,-24,-24,-36,-36])   #纬度
    x5, y5 = m(x5,y5)
    m.plot(x5,y5,color='b',linewidth=1)
    '''

    #parallels = np.arange(0.,80,20.)
    #m.drawparallels(parallels,labels=[0,0,1,1])
    #meridians = np.arange(10.,360.,20.)
    #m.drawmeridians(meridians,labels=[1,1,1,1])

    plt.title(ptitle,y=1.075,size=6)#, fontproperties='SimHei')
    print 'plotting with basemap ...'
    fig.savefig(imgfile, dpi=180)
    #fig.savefig('a.pdf', dpi=180)
    #fig.savefig(imgfile.replace('.png','.pdf'))
    #fig.savefig(imgfile.replace('.png','.svg'))
    #fig.savefig(imgfile.replace('.png','.eps'))
    df.CutPicWhiteBorder(imgfile)
    if(showimg):
        plt.show()
    plt.close()
