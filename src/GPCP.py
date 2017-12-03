# coding:utf-8
# GPCP.py: 检验GPCP资料自1979年以来格点的逐月降水是否满足正态分布,统计并绘图
# author:qkx/kxqiu@chinkun.cn
# latest:2016.10.18

import netCDF4
from mpl_toolkits.basemap import Basemap,cm
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Arial Black'
import numpy as np
from matplotlib.animation import FuncAnimation
import datetime as dt
import matplotlib.mlab as mlab
import math
from scipy.optimize import curve_fit

STORE_PATH = '../tmp/typhoon'

fh = netCDF4.Dataset('../tmp/CSM_PREC_07.nc','r')

prob = fh.variables['prob'][:]
obs = fh.variables['obs'][:]
time = fh.variables['times'][:]
# d1 =  prob[:,4,129,295]*86400000
# d2 =  obs[:,1,129,295]*86400000
# lat = fh.variables['lat'][:]
# lon = fh.variables['lon'][:]
# plt.plot(time,d1)
# plt.plot(time,d2,'r')
# plt.show()
fh_obs = netCDF4.Dataset('../tmp/precip.mon.mean.nc','r')
x = fh_obs.variables['time'][:]
y = fh_obs.variables['precip'][:,52,48]
mu = np.mean(y)
var = np.var(y)
xx = np.linspace(0,mu + 3,100)

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

## plot and legend
n,bins,patches = plt.hist(y,50,normed=True,range=(0,5),alpha=0.5,label='histogram')
p0 = [1., 0., 1.]
bin_centers = (bins[:-1] + bins[1:])/2
coeff, var_matrix = curve_fit(gauss, bin_centers, n, p0=p0)

# Get the fitted curve
hist_fit = gauss(bin_centers, *coeff)

plt.plot(xx,mlab.normpdf(xx,mu,math.sqrt(var)),color='r',label='calculate')
plt.plot(bin_centers,hist_fit,'g--',label='curve_fit')
plt.legend([u'Normal Distribution(均值:{:.2f},方差:{:.2f})'.format(mu,var),\
            u'曲线拟合(均值:{:.2f},方差:{:.2f})'.format(coeff[1],coeff[2]),\
            u'归一化的降水统计频次'])

## title and label
plt.title('GPCP monthly precipitation dataset Normalization Fit')
plt.xlabel('precipitation(mm/day)')
plt.ylabel('Probability')

plt.show()
#
# plt.savefig('gpcp.png')