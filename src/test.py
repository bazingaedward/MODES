# coding : utf-8
"""
    hello
"""
import numpy as np
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt
# import datetime as dt
# import matplotlib.dates as mdates
# import subprocess
# from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
# import os

# fh = netCDF4.Dataset('../tmp/hgt.1971.nc','r')
# time = fh.variables['time']

names = ['year','month','day','area','intensity','lines','points']
df_cal = pd.read_csv('~/modestmp/SH.1971.CAL.txt',header=None,names=names,delim_whitespace=True,na_values=-999,index_col=None)
df_compare = pd.read_csv('~/modestmp/SH_1971.txt',header=None,names=names,delim_whitespace=True,na_values=-999,index_col=None)
# df = df.replace(-999,np.nan)
times = pd.to_datetime(df_cal[['year','month','day']])

var = 'points'
plt.clf()
ax = plt.subplot(111)
ax.plot(times, df_compare[var],'-',color='r',alpha=0.8)
ax.plot(times, df_cal[var],'-',alpha=0.5)
ax.xaxis_date()
plt.show()


# import netCDF4
# from mpl_toolkits.basemap import Basemap,cm
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.animation import FuncAnimation
# import datetime as dt
#
# STORE_PATH = '../tmp/typhoon'
#
# fh = netCDF4.Dataset('/Users/qiukaixiang/Desktop/test.nc','r')
#
# # print fh.variables
# time = fh.variables['time']
# lat = fh.variables['lat'][:]
# lon = fh.variables['lon'][:]
# data = fh.variables['rv'][:]
#
#
# fig, ax = plt.subplots()
# fig.set_tight_layout(True)
#
# map = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=80,\
#             llcrnrlon=60,urcrnrlon=200,resolution='c')
# map.drawcoastlines(linewidth=0.25)
# map.drawcountries(linewidth=0.25)
# # map.fillcontinents(color='coral',lake_color='aqua')
# # map.drawmapboundary(fill_color='aqua')
# lons, lats = np.meshgrid(lon,lat)
# x, y = map(lons, lats)
# # print x,y
# cs = map.contourf(x,y,data[0,:,:]*100000,cmap=plt.cm.jet)
# cbar = map.colorbar(cs,location='bottom',pad="5%")
# start = dt.datetime(2012,1,1)
# end = dt.datetime(2012,8,23)
#
# def update(i):
#     title = ' '.join(['Relative Vorticity(Western Pacific Ocean)',(end + dt.timedelta(hours=6*i)).strftime("%Y-%m-%d %H")])
#     cs = map.contourf(x,y,data[4*(end-start).days+i,:,:]*100000,cmap=plt.cm.jet)
#     ax.set_title(title)
#     return cs
#
# if __name__ == '__main__':
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 28), interval=20)
#     # anim.save('blw.gif', dpi=160, writer='imagemagick')
#     plt.show()

# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# fig, ax = plt.subplots()
# fig.set_tight_layout(True)
#
#
# print('fig size: {0} DPI, size in inches {1}'.format(
#     fig.get_dpi(), fig.get_size_inches()))
#
# x = np.arange(0, 20, 0.1)
# ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
# line, = ax.plot(x, x - 5, 'r-', linewidth=2)

# def update(i):
#     label = 'timestep {0}'.format(i)
#     print(label)
#     line.set_ydata(x - 5 + i)
#     ax.set_xlabel(label)
#     return line, ax
#
# if __name__ == '__main__':
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
#     if len(sys.argv) > 1 and sys.argv[1] == 'save':
#         anim.save('line.gif', dpi=80, writer='imagemagick')
#     else:
#         plt.show()