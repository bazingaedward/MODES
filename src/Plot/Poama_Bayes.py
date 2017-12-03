# coding:utf-8
# author:qkx<kxqiu@chinkun.cn>
# latest:2017/9/27


import netCDF4
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import glob



def plot(x, y, zi, title):
    map = Basemap(projection='gall',resolution='l',llcrnrlon=40,llcrnrlat=-10,urcrnrlon=160,urcrnrlat=61)
    map.drawcoastlines()
    map.drawcountries()
    parallels = np.arange(-10.,60,15.)
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=8)
    meridians = np.arange(40.,160.,20.)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8)
    # map.drawstates()
    x1, y1 = np.meshgrid(x, y)
    xx, yy = map(x1, y1)

    cs = map.contourf(xx, yy, zi, cmap='YlGnBu')
    cbar = map.colorbar(cs,location='bottom',pad="8%")
    cbar.set_label('mm')
    plt.title(title)
    plt.savefig(title + '.png')
    # plt.show()


if __name__ == '__main__':
    fh = netCDF4.Dataset('../mac_20170924.nc')
    x = fh.variables['lon'][:]
    y = fh.variables['lat'][:]
    zi = fh.variables['hr24_prcp'][1, :]
    print fh.variables['time'][:]
    fh.close()
    plot(x, y, zi, 'MODES_POAMA_20170920_PREC_Bayes_After')