# -*- coding: cp936 -*-
from __future__ import print_function
import sys,os,re,gc,time
from HTMLParser import HTMLParser
import numpy as np
import netCDF4 as nc4
from dateutil.relativedelta import relativedelta
import time as getsystime
import timeit
import time
from mpl_toolkits.basemap import shapefile
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
#增加一个月的计算
def add_one_month(t):
    """Return a `datetime.date` or `datetime.datetime` (as given) that is
    one month earlier.
    
    Note that the resultant day of the month might change if the following
    month has fewer days:

        >>> add_one_month(datetime.date(2010, 1, 31))
        datetime.date(2010, 2, 28)
    """
    import datetime
    one_day = datetime.timedelta(days=1)
    one_month_later = t + one_day
    while one_month_later.month == t.month:  # advance to start of next month
        one_month_later += one_day
    target_month = one_month_later.month
    while one_month_later.day < t.day:  # advance to appropriate day
        one_month_later += one_day
        if one_month_later.month != target_month:  # gone too far
            one_month_later -= one_day
            break
    return one_month_later

#------------------------------------------------------------------------------
#减少一个月的计算
def subtract_one_month(t):
    """Return a `datetime.date` or `datetime.datetime` (as given) that is
    one month later.
    
    Note that the resultant day of the month might change if the following
    month has fewer days:
    
        >>> subtract_one_month(datetime.date(2010, 3, 31))
        datetime.date(2010, 2, 28)
    """
    import datetime
    one_day = datetime.timedelta(days=1)
    one_month_earlier = t - one_day
    while one_month_earlier.month == t.month or one_month_earlier.day > t.day:
        one_month_earlier -= one_day
    return one_month_earlier
#-------------------------------------------------------------------------------
#分析网页链接的类    
#hp = df.MyHTMLParser()
#    hp.feed(data)
#    hp.close()
#    for link in hp.links:
#        print link
#        for str1 in list_datestr:
class MyHTMLParser(HTMLParser):  
    def __init__(self):  
        HTMLParser.__init__(self)  
        self.flag = 0  
        self.links = []  
        self.title=""  
        self.img=""  
        self.content=""  
   
    def handle_starttag(self, tag, attrs):  
        #print "Encountered the beginning of a %s tag" % tag  
        if tag == "a":  
            if len(attrs) == 0: pass  
            else:  
                for (variable, value)  in attrs:  
                    if variable == "href":  
                        self.links.append(value)      

#------------------------------------------------------------------------------
def get_ncep_grib_data(FName,levels_num):
    """
    获取NCEP格式的grib导出数据
    """
    if(not os.path.isfile(FName)):
        return os.path.isfile(FName)
    Lx=73;Ly=144
    #FName = 'allprs.bin'
    #ux = np.fromfile(FName, count=Lx*Ly*17, dtype=(np.float32))
    #ua = ux.reshape(17,Lx,Ly)
    #ua = ua.transpose()
    #ua = ux.reshape(17,Lx,Ly);
    #np.transpose(x, (1, 0, 2)).shape
    #ua = np.transpose(ua,(2,1,0))
    #print ux.shape,ua.shape
    #return ua;
    #sys.exit(0)
    str_dtype=">(%d)f"%(Lx*Ly)
    str_dtype= (str_dtype+',')*(levels_num-1)+str_dtype

    #str_dtype = np.dtype(('>f4,>(2)f4,>f4', (4)))
    #print str_dtype
    Prs=np.zeros((levels_num,Lx,Ly))
    ux = np.fromfile(FName,dtype=str_dtype);
    for i in range(0,levels_num):
        Prs[i,:,:]  = np.reshape(ux[0][i],(Lx,Ly))
        print(i)

    return Prs
    #ua=ux[0][0]
    #ua=ux[0][1]
    #ua=ux[0][16]
    #ua=ua.reshape(Lx,Ly)

    #lon=np.arange(0,360,2.5,dtype=float);
    #lat=np.arange(90,-90-1,-2.5,dtype=float);
    #print lon.shape,lat.shape
    #dplot.drawhigh(ua,lon,lat,ptype=1,ptitle='aaa',imgfile='Field1.png',showimg=1)

#------------------------------------------------------------------------------
def getnc_max_date(FieldFileName):
    rootgrp = nc4.Dataset(FieldFileName,'r')
    #print rootgrp.file_format
    #print rootgrp.variables
    #print('-'*80)
    times = rootgrp.variables['time'];
    dates = nc4.num2date(times[:],units=times.units)
    rootgrp.close()
    return dates[-1];

#-----------------------------------------------------------`-------------------
def get_dir_list(dirname, re_str):
    '''
    #m = re.search(r'hello', 'hello world!')
    #print m
    '''
    #print(dirname)
    flist = []
    for dirpath, dirnames, filenames in os.walk(dirname):
        #print(filenames)
        for filename in filenames:
            print(filename)
            #if os.path.splitext(filename)[1] == '.grb2':
            filepath = os.path.join(dirpath, filename)
            if(re.search(re_str, filepath)):
                flist.append(filepath)
    flist.sort()
    return flist


################################################################################
def create_climate_model_nc(NcFileName, list2,var_name='hgt',ncformat='NETCDF3_CLASSIC',att_NcFile={},att_var={}):
    '''生成数值模式用nc文件'''

    #获取List中的最小和最大日期 date1,date2

    date1 = list2[0][0]
    date2 = list2[-1][0]

    print('step01',date1, date2)
    for i in  range(len(list2)):
        if date1>list2[i][0]:
            date1=list2[i][0]

    for i in range(len(list2)):
        if date2<list2[i][0]:
            date2=list2[i][0]
    #结束获取List中的最小和最大日期 date1,date2
    print('step02',date1, date2)
    from  datetime import datetime
    s_date1 = datetime.strftime(date1,'%Y-%m-01')
    s_date2 = datetime.strftime(date2,'%Y-%m-01')
    date1 = datetime.strptime(s_date1,'%Y-%m-%d')
    date2 = datetime.strptime(s_date2,'%Y-%m-%d')


    #print (list2[-1][0]-list2[0][0]).hour
    r = relativedelta(date2, date1)
    months_count = r.years * 12 + r.months+1
    print( 'month_count=',months_count)

    date_list = []
    ii=0
    #while date1 <= date2:
    while ii < months_count:
        date_list.append(date1)
        date1 = date1 + relativedelta(months=1)
        ii=ii+1
        #print date1
    print(len(date_list),'=?',months_count)
    #sys.exit(0)
    print('step03',date1, date2)
    #print('date_list=',date_list)

    #------------------------------------------------------------------------------
    rootgrp = nc4.Dataset(NcFileName, 'w', format=ncformat) #NETCDF4_CLASSIC
    print(rootgrp.file_format)
    time = rootgrp.createDimension('time', None)
    #level = rootgrp.createDimension('level', 1)
    lat = rootgrp.createDimension('lat', 73)
    lon = rootgrp.createDimension('lon', 144)

    times = rootgrp.createVariable('time', 'f8', ('time',))
    #levels = rootgrp.createVariable('level', 'i4', ('level',))
    latitudes = rootgrp.createVariable('lat', 'f4', ('lat',))
    longitudes = rootgrp.createVariable('lon', 'f4', ('lon',))

    #NETCDF3_CLASSIC
    if 'NETCDF3_CLASSIC'== ncformat :
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',))#,zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time','lat', 'lon',))#,zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True

    if 'NETCDF4_CLASSIC'== ncformat :
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'lat', 'lon',),zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True

    #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True
    #dplot.drawhigh(var1,lon,lat,ptype=1,ptitle='aaa',imgfile='Field1.png',showimg=1)
    #raw_input()

    rootgrp.CreatedDateTime = 'Created ' + getsystime.ctime(getsystime.time())

    #给NC文件添加属性
    for key in att_NcFile.keys():
        print(key, '\t', att_NcFile[key])
        setattr(rootgrp,key,att_NcFile[key])

    for name in rootgrp.ncattrs():
        print( 'Global attr', name, '=', getattr(rootgrp,name) )
        
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    #levels.units = 'hPa'

    #给变量增加属性
    for key in att_var.keys():
        print(key, '\t', att_var[key])
        setattr(hgt_500mb,key,att_var[key])

    #hgt_500mb.units = 'hPa'
    #hgt_500mb.level = '500hPa'
    #hgt_500mb.long_name = 'Monthly mean geopotential height'

    print('*'*80)
    print(type(hgt_500mb))

    times.units = 'hours since 0001-01-01 00:00:00.0'
    times.calendar = 'gregorian'

    #for name in rootgrp.ncattrs():
    #    print('Global attr', name, '=', getattr(rootgrp, name))

    longitudes[:] = np.arange(0, 360, 2.5, dtype=float)
    latitudes[:] = np.arange(90, -90 - 1, -2.5, dtype=float)
    #levels[:] = [500]
    #hgt_500mb[:, :, :, :] = np.zeros((months_count, 1, 73, 144), dtype=np.float)
    hgt_500mb[:, :, :] = np.zeros((months_count, 73, 144), dtype=np.float)

    #print hgt_h500.shape[0]

    #sys.exit(0)
    #print lon.shape,lat.shape
    print('-'*80)
    print('list date')
    for list1 in list2:
        print(datetime.strftime(list1[0],'%Y%m%d'),end=' ')
    print('\n','-'*80)

    time_num_list = []
    i = 0
    for date1 in date_list:
        #print(date1)
        time_num_list.append(nc4.date2num(date1, units=times.units, calendar=times.calendar))
        for list1 in list2:
            #print(list1[0])      print(type(date1))         print(type(list1[0]))
            #if(date1 == list1[0]):
            if(datetime.strftime(date1,'%Y-%m')==datetime.strftime(list1[0],'%Y-%m')):
                #print(datetime.strftime(date1,'%Y-%m'))
                print('%4d'%i,end=' ')#,date1,list1[0]) #end=' '

                hgt_500mb[i, :, :] = list1[2]
                #print np.shape(list1[2])

        i = i + 1
        if((i+1)%12==0):print('')

    #print time_num_list
    #print range(348)
    times[:] = time_num_list
    #times[:] = np.asarray(range(348))
    #nc4.date2num(dates,units=times.units)
    #del flist1
    print('')
    rootgrp.close()
    #结束生成


################################################################################
def create_climate_model_nc_old(NcFileName, list2,var_name='hgt',ncformat='NETCDF3_CLASSIC'):
    '''生成数值模式用nc文件'''

    #获取List中的最小和最大日期 date1,date2
    date1 = list2[0][0]
    date2 = list2[-1][0]
    print(date1, date2)
    for i in  range(len(list2)):
        if date1>list2[i][0]:
            date1=list2[i][0]

    for i in range(len(list2)):
        if date2<list2[i][0]:
            date2=list2[i][0]

    print(date1, date2)

    #print (list2[-1][0]-list2[0][0]).hour
    r = relativedelta(date2, date1)
    months_count = r.years * 12 + r.months+1
    print( months_count)

    date_list = []

    ii=0
    #while date1 <= date2:
    while(ii<months_count ):
        date_list.append(date1)
        date1 = date1 + relativedelta(months=1)
        ii=ii+1
        #print date1
    print(len(date_list),months_count)
    sys.exit(0)
    print(date1, date2)

    #------------------------------------------------------------------------------
    rootgrp = nc4.Dataset(NcFileName, 'w', format=ncformat) #NETCDF4_CLASSIC
    print(rootgrp.file_format)
    time = rootgrp.createDimension('time', None)
    level = rootgrp.createDimension('level', 1)
    lat = rootgrp.createDimension('lat', 73)
    lon = rootgrp.createDimension('lon', 144)

    times = rootgrp.createVariable('time', 'f8', ('time',))
    levels = rootgrp.createVariable('level', 'i4', ('level',))
    latitudes = rootgrp.createVariable('lat', 'f4', ('lat',))
    longitudes = rootgrp.createVariable('lon', 'f4', ('lon',))

    #NETCDF3_CLASSIC
    if 'NETCDF3_CLASSIC'== ncformat :
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',))#,zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time','lat', 'lon',))#,zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True

    if 'NETCDF4_CLASSIC'== ncformat :
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'lat', 'lon',),zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True

    #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9 )#,least_significant_digit=1) #,zlib=True
    #dplot.drawhigh(var1,lon,lat,ptype=1,ptitle='aaa',imgfile='Field1.png',showimg=1)
    #raw_input()
    rootgrp.description = 'CFS DATA File'
    rootgrp.history = 'Created ' + getsystime.ctime(getsystime.time())
    rootgrp.source = 'netCDF4 python module tutorial'
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    levels.units = 'hPa'
    hgt_500mb.units = 'gpm'

    times.units = 'hours since 0001-01-01 00:00:00.0'
    times.calendar = 'gregorian'

    for name in rootgrp.ncattrs():
        print('Global attr', name, '=', getattr(rootgrp, name))

    longitudes[:] = np.arange(0, 360, 2.5, dtype=float)
    latitudes[:] = np.arange(90, -90 - 1, -2.5, dtype=float)
    levels[:] = [500]
    hgt_500mb[:, :, :, :] = np.zeros((months_count, 1, 73, 144), dtype=np.float)
    #hgt_500mb[:, :, :] = np.zeros((months_count, 73, 144), dtype=np.float)

    #print hgt_h500.shape[0]

    #sys.exit(0)
    #print lon.shape,lat.shape


    time_num_list = []
    i = 0
    for date1 in date_list:
        #print date1
        time_num_list.append(nc4.date2num(date1, units=times.units, calendar=times.calendar))
        for list1 in list2:
            if(date1 == list1[0]):
                print(i,end='')
                hgt_500mb[i, 0,:, :] = list1[2]
                #print np.shape(list1[2])

        i = i + 1

    #print time_num_list
    #print range(348)
    times[:] = time_num_list
    #times[:] = np.asarray(range(348))
    #nc4.date2num(dates,units=times.units)
    #del flist1
    print('')
    rootgrp.close()
    #结束生成

def append_climate_model_nc(NcFileName, list2,var_name='hgt',att_var={}):
    '''对模式生成的nc文件添加变量'''
    from  datetime import datetime
    date1 = list2[0][0]
    date2 = list2[-1][0]
    print(date1, date2)

    #print (list2[-1][0]-list2[0][0]).hour
    r = relativedelta(date2, date1)
    months_count = r.years * 12 + r.months
    print(months_count)

    #date_list = []
    #while date1 <= date2:
    #    date_list.append(date1)
    #    date1 = date1 + relativedelta(months=1)
        #print date1

    print(date1, date2)

    #------------------------------------------------------------------------------
    rootgrp = nc4.Dataset(NcFileName, 'a') #NETCDF4_CLASSIC 'a'指append
    times = rootgrp.variables['time'];
    
    print(times)
    date_list = nc4.num2date(times,times.units)
    print(date_list)
    

    print(rootgrp.file_format)

    if('NETCDF3_CLASSIC'==rootgrp.file_format):
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',))
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'lat', 'lon',))

    if('NETCDF4_CLASSIC'==rootgrp.file_format):
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9)
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'lat', 'lon',),zlib=True, complevel=9)

    #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9)#,least_significant_digit=1) #,zlib=True
    #dplot.drawhigh(var1,lon,lat,ptype=1,ptitle='aaa',imgfile='Field1.png',showimg=1)
    #raw_input()
    #hgt_500mb.units = 'gpm'
    for key in att_var.keys():
        print(key, '\t', att_var[key])
        setattr(hgt_500mb,key,att_var[key])

    
    for name in rootgrp.ncattrs():
        print('Global attr', name, '=', getattr(rootgrp, name))


    #hgt_500mb[:, :, :, :] = np.zeros((months_count, 1, 73, 144), dtype=np.float)
    hgt_500mb[:, :, :] = np.zeros((len(date_list), 73, 144), dtype=np.float)

    #print hgt_h500.shape[0]


    #sys.exit(0)
    #print lon.shape,lat.shape


    time_num_list = []
    i = 0
    for date1 in date_list:
        #print date1
        #time_num_list.append(nc4.date2num(date1, units=times.units, calendar=times.calendar))
        for list1 in list2:
            #if(date1 == list1[0]):
            if(datetime.strftime(date1,'%Y-%m')==datetime.strftime(list1[0],'%Y-%m')):
                print('%4d'%i,end=' ')
                hgt_500mb[i, :, :] = list1[2]
                #print np.shape(list1[2])

        i = i + 1
        if((i+1)%12==0):print('')
    #print time_num_list
    #print range(348)
    #times[:] = time_num_list
    #times[:] = np.asarray(range(348))
    #nc4.date2num(dates,units=times.units)
    #del flist1
    print('')
    rootgrp.close()

def append_climate_model_nc_old(NcFileName, list2,var_name='hgt'):
    '''对模式生成的nc文件添加变量'''
    date1 = list2[0][0]
    date2 = list2[-1][0]
    print(date1, date2)

    #print (list2[-1][0]-list2[0][0]).hour
    r = relativedelta(date2, date1)
    months_count = r.years * 12 + r.months
    print(months_count)

    date_list = []
    while date1 <= date2:
        date_list.append(date1)
        date1 = date1 + relativedelta(months=1)
        #print date1

    print(date1, date2)

    #------------------------------------------------------------------------------
    rootgrp = nc4.Dataset(NcFileName, 'a') #NETCDF4_CLASSIC 'a'指append
    print(rootgrp.file_format)

    if('NETCDF3_CLASSIC'==rootgrp.file_format):
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',))
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'lat', 'lon',))

    if('NETCDF4_CLASSIC'==rootgrp.file_format):
        hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9)
        #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'lat', 'lon',),zlib=True, complevel=9)

    #hgt_500mb = rootgrp.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon',),zlib=True, complevel=9)#,least_significant_digit=1) #,zlib=True
    #dplot.drawhigh(var1,lon,lat,ptype=1,ptitle='aaa',imgfile='Field1.png',showimg=1)
    #raw_input()
    #hgt_500mb.units = 'gpm'

    for name in rootgrp.ncattrs():
        print('Global attr', name, '=', getattr(rootgrp, name))


    hgt_500mb[:, :, :, :] = np.zeros((months_count, 1, 73, 144), dtype=np.float)
    #hgt_500mb[:, :, :] = np.zeros((months_count, 73, 144), dtype=np.float)

    #print hgt_h500.shape[0]


    #sys.exit(0)
    #print lon.shape,lat.shape


    time_num_list = []
    i = 0
    for date1 in date_list:
        #print date1
        #time_num_list.append(nc4.date2num(date1, units=times.units, calendar=times.calendar))
        for list1 in list2:
            if(date1 == list1[0]):
                print(i,end=' ')
                hgt_500mb[i,0, :, :] = list1[2]
                #print np.shape(list1[2])

        i = i + 1

    #print time_num_list
    #print range(348)
    #times[:] = time_num_list
    #times[:] = np.asarray(range(348))
    #nc4.date2num(dates,units=times.units)
    #del flist1
    print('')
    rootgrp.close()
#------------------------------------------------------------------------------

def grid_to_144x73(lat1,lon1,zi,**args):
    '''
    将高斯格点等资料插值到144x73
    '''
    lon2 = args.pop("lon", np.arange(0, 360, 2.5, dtype=float) )
    lat2 = args.pop("lat", np.arange(90,-90-1,-2.5,dtype=float) )
    from matplotlib.mlab import griddata

    xi, yi = np.meshgrid(lon1,lat1)
    #zi = np.loadtxt('a.txt')

    #lon2 = np.arange(0, 360, 2.5, dtype=float)
    #lat2 = np.arange(90,-90-1,-2.5,dtype=float)

    #points = np.vstack((x2,y2)).T
    ##lat = np.arange(-90, 90+1, 2.5, dtype=float)

    #print('*'*40)
    #print xi.shape,yi.shape,zi.shape,lon2.shape,lat2
    zi = griddata(xi.flatten(),yi.flatten(),zi.flatten(),lon2,lat2)#,interp='linear')
    zi[0]=np.mean(zi[1])
    zi[-1]=np.mean(zi[-2])
    print('.'),
    return zi

#------------------------------------------------------------------------------
def griddata_scipy_idw(x, y, z, xi, yi,function='cubic'):
    '''
    scipy反向距离加权插值
    'multiquadric': sqrt((r/self.epsilon)**2 + 1)  #不能
    'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1) #不能
    'gaussian': exp(-(r/self.epsilon)**2) 不能用来插值
    'linear': r  #能
    'cubic': r**3 #能
    'quintic': r**5  #效果差，勉强能
    'thin_plate': r**2 * log(r)  能可以用用来插值
    '''
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

    (nx,ny)=xi.shape
    xi, yi = xi.flatten(), yi.flatten()
    from scipy.interpolate import Rbf
    interp = Rbf(x, y, z, function=function,epsilon=1)#linear
    zi = np.reshape(interp(xi, yi),(nx,ny))
    zi = zi.astype(np.float32)
    return zi


#------------------------------------------------------------------------------
def griddata_linear_rbf(x, y, z, xi, yi):
    '''
    离散点插值为网格点的函数，速度较快
    网格点上限不要超过500x500会溢出
    '''
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)
    (nx,ny)=xi.shape
    xi, yi = xi.flatten(), yi.flatten()


    dist = distance_matrix(x,y, xi,yi)
    #print 'dist shape =',dist.shape,dist.dtype

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y,x,y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)
    #print(weights.dtype)
    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    #print(xi.shape,yi.shape, zi.shape)
    zi = zi.reshape(nx,ny)
    zi = zi.astype(np.float32)
    return zi


#------------------------------------------------------------------------------
def griddata_linear_rbf2(x, y, z, xi, yi,function='linear'):

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

    (nx,ny)=xi.shape
    xi, yi = xi.flatten(), yi.flatten()
    from scipy.interpolate import Rbf
    interp = Rbf(x, y, z, epsilon=1)#linear
    zi = np.reshape(interp(xi, yi),(nx,ny))
    zi = zi.astype(np.float32)
    return zi



def distance_matrix(x0, y0, x1, y1):
    '''距离矩阵'''
    
    x0= x0.astype(np.float32)
    y0= y0.astype(np.float32)
    x1= x1.astype(np.float32)
    y1= y1.astype(np.float32)
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    #print 'hypot d0,d1',d0.shape,d1.shape,d0.dtype,d1.dtype
    return np.hypot(d0, d1)

#...............................................................................
class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        from scipy.spatial import cKDTree as KDTree
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]

#------------------------------------------------------------------------------
def griddata_Invdisttree(x, y, z, xi, yi,**args):
    '''
        反向距离插值
    '''
    Nnear = args.pop("Nnear", 8)   #Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = args.pop("leafsize", 10)  #    leafsize = 10
    eps = args.pop("eps",0.1) # eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = args.pop("p",1) #  p = 1  # weights ~ 1 / distance**p

    (nx,ny)=xi.shape
    xi, yi = xi.flatten(), yi.flatten()
    
    obsxy = np.column_stack((x,y))
    askxy = np.column_stack((xi,yi))

    print(obsxy.shape,z.shape)

    invdisttree = Invdisttree( obsxy, z, leafsize=leafsize, stat=1 )
    print('invdisttree.shape=',invdisttree)
    interpol = invdisttree( askxy, nnear=Nnear, eps=eps, p=p )
    print('interpol.shape=',interpol.shape)

    interpol = interpol.reshape(nx,ny)
    return interpol


#------------------------------------------------------------------------------

def griddata_all(x,y,z,x1,y1,func='line_rbf'):
    '''
    把各种插值方法整合到一起
    scipy_idw
    line_rbf
    Invdisttree
    nat_grid
    '''
    xi, yi = np.meshgrid(x1, y1)

    if('griddata'==func):
        from matplotlib.mlab import griddata
        zi = griddata(x,y,z,x1,y1)


    if('scipy_idw'==func):
        zi= griddata_scipy_idw(x,y,z,xi,yi)

    if('line_rbf'==func):
        zi = griddata_linear_rbf(x,y,z,xi,yi)  #        grid3 = grid3.reshape((ny, nx))

    if('line_rbf2'==func):
        zi = griddata_linear_rbf2(x,y,z,xi,yi)  #        grid3 = grid3.reshape((ny, nx))


    if('Invdisttree'==func):
        #zi = df.griddata_Invdisttree(x,y,z,xi,yi,Nnear=15,p=3,eps=1)
        zi = griddata_Invdisttree(x,y,z,xi,yi,p=3)#,Nnear=10,eps=1)

    if('nat_grid'==func):
        from griddata import griddata, __version__
        zi = griddata(x,y,z,xi,yi)

    #if('test'==func):
    #    zi = griddata_scipy_spatial(x,y,z,xi,yi)

    return zi,xi,yi


#------------------------------------------------------------------------------
def extened_grid(zi,x1,y1,zoom=2):
    '''
    xinterval : X插值的间隔
    yinterval : Y 插值的间隔
    扩展网格区域zoom为扩展倍数
    '''
    #print(x1)
    nx = np.size(x1)
    ny = np.size(y1)
    x2 = np.linspace(x1.min(), x1.max(), nx * zoom)
    y2 = np.linspace(y1.min(), y1.max(), ny * zoom)
    xi,yi = np.meshgrid(x2,y2)

    #插值方法1 Zoom方法
    #from scipy import ndimage
    #z2 = ndimage.interpolation.zoom(zi[:,:], zoom)

    #插值方法2 basemap.interp方法
    from mpl_toolkits.basemap import interp
    z2 = interp(zi, x1, y1, xi, yi, checkbounds=True, masked=False, order=1)

    #插值方法3 interpolate.RectBivariateSpline 矩形网格上的样条逼近。
    # Bivariate spline approximation over a rectangular mesh
    #from scipy import interpolate
    #sp = interpolate.RectBivariateSpline(y1,x1,zi,kx=1, ky=1, s=0)
    #z2 = sp(y2,x2)



    


    #sp = interpolate.LSQBivariateSpline(y1,x1,zi)
    #z2 = sp(y2,x2)

    #terpolate.LSQBivariateSpline?

    print('extend shapes:=',z2.shape,xi.shape,yi.shape)
    return z2,xi,yi,x2,y2,nx*zoom,ny*zoom
    #print(x3)


#------------------------------------------------------------------------------
def draw_map_lines(m,shapefilename,color='k',linewidth =0.2,debug=0):
    '''
    m:basemap对象
    shapefilename：为shapefile的名称
    '''
    sf = shapefile.Reader(shapefilename)
    shapes = sf.shapes()
    i=0
    for shp in shapes:
        #print(i,end=' ') # i=i+1
        xy = np.array(shp.points)
        if(1==debug):
            #z4 = np.vstack((x,y4))
            #print(z4.shape)
            np.savetxt('n%d.txt'%i,xy,fmt='%6.2f')

        x4,y4=m(xy[:,0],xy[:,1])
        m.plot(x4,y4,color=color,linewidth=linewidth)
        #if(i>0):
        #    break
        

#------------------------------------------------------------------------------
def build_inside_mask_array(shapefilename,x1,y1):    #    r"spatialdat\china_province"
    '''
    #2011-11-12
    #利用给点的shp文件构造一个可以用来mask制定区域的数组
    #----------------------------------------------------------------------
    #构造一个数组，全为否，用于只画中国区域的图
    '''
    from mpl_toolkits.basemap import shapefile
    from matplotlib.nxutils import points_inside_poly
    xi, yi = np.meshgrid(x1, y1)
    grid1 = np.ones_like(xi)
    grid1 = grid1<0
    #grid1 = grid1.flatten()
    sf = shapefile.Reader(shapefilename)
    shapes = sf.shapes()
    for shp in shapes:
        #see http://code.google.com/p/pyshp/
        #print(shp.bbox)
        srows = np.logical_and(x1>shp.bbox[0],x1<shp.bbox[2])
        scols = np.logical_and(y1>shp.bbox[1],y1<shp.bbox[3])
        selgrid =np.dot(np.atleast_2d(srows).T,np.atleast_2d(scols)).T
        #print(selgrid.shape,xi.shape,yi.shape,grid1.shape)

        points = np.vstack((xi[selgrid].flatten(),yi[selgrid].flatten())).T
        grid2 = points_inside_poly(points, shp.points)
        grid1[selgrid] = np.logical_or(grid2,grid1[selgrid])
    #sys.exit(0)
    #----------------------------------------------------------------------
    return  grid1,shapes
#------------------------------------------------------------------------------
def mytic():
    '''
    记时函数,与mytoc成对出现,充分利用了List的Append 和 pop函数
    '''
    global starttime
    if 'starttime' in globals():
        starttime.append(time.clock())
    else:
        starttime=[]
        starttime.append(time.clock())

#------------------------------------------------------------------------------

def mytoc(str1=''):
    '''
    记时函数,与mytoc成对出现
    '''
    global starttime
    endtime = getsystime.clock()
    if len(starttime)>0:
        start=starttime.pop()
        print('%s cost %8.2f seconds' % (str1,endtime - start))
    else:
        print('mytoc error!')
#------------------------------------------------------------------------------

def Read_Ncep_Hgt(NcFileName,I_Year,Mon,Months_Count,FieldOffset=0,var_name='hgt',ilev=5):
    #FieldFileName = r'd:\ncepmon\hgt.mon.mean.nc'
    """
        读取ncep 高度场信息，同时也包括风场和温度场等

        FieldOffset 指往前偏移的月份
    """
    print('FieldOffset=%d'%(FieldOffset))
    print('Months_Count=%d'%(Months_Count))
    Months_Count = Months_Count
    
    rootgrp = nc4.Dataset(NcFileName,'r')
    print(NcFileName+"  Format is: "+rootgrp.file_format)
    print(rootgrp.variables)

    dinfo={}

    print('#'+'-'*79)
    print('read %s 数据'%(NcFileName))
    print('lev=',ilev,type(ilev))

    lat = rootgrp.variables['lat'][:]
    lon = rootgrp.variables['lon'][:]
    
    if 'level' in rootgrp.variables:
        level = rootgrp.variables['level'][:]
        dinfo["level"]=int(level[ilev])
    else:
        print("can't find level variable")

    times = rootgrp.variables['time']
    hgt = rootgrp.variables[var_name]

    #print(var_name+' shape is:',end='')  print(hgt.shape)   print(type(hgt)) sys.exit(0)

    dates = nc4.num2date(times[:],units=times.units)
    from  datetime import datetime

    MaxDate = datetime.strptime('%04d-%02d-01'%(max(I_Year),Mon),'%Y-%m-%d')+relativedelta(months=Months_Count-1-FieldOffset)
    #print(type( MaxDate) ) #print(type( dates.max()) )

    if(dates.max() < MaxDate ):
        print('dates.max() is',dates.max())
        print('MaxDate is',MaxDate)
        print('访问时间的上限已经超过Netcdf文件时间维的上限')

        sys.exit(0)

    Field = np.zeros( (len(I_Year),len(lat),len(lon)) )
    j=0
    for i in range(len(dates)):
        datestr_c =  datetime.strftime(dates[i],'%Y-%m')
        #print(datestr_c)
        for year1 in I_Year:
            s1 ='%04d-%02d'%(year1,Mon)
            #date1 = datetime.strptime(s1,'%Y-%m-%d')
            #date2 = date1 + relativedelta(months=Months_Count-1)
            #print(date1)
            if s1==datestr_c:
                L1 = i
                L2 = L1+Months_Count
                L1 = L1 - FieldOffset  #减去往前偏移的月数
                L2 = L2 - FieldOffset  #减去往前偏移的月数

                date1 = datetime.strptime(s1,'%Y-%m')
                date1 = date1 + relativedelta(months=-FieldOffset)
                dinfo["mon1"]='%s'%int( datetime.strftime(date1,'%m') )
                date1 = date1 + relativedelta(months=Months_Count-1)
                dinfo["mon2"]='%s'%int( datetime.strftime(date1,'%m') )
                print(u'月份'+ dinfo["mon1"]+ dinfo["mon2"],end=' ')

                print('%02d'%j,end=' ')
                print(s1,'L1=',L1,'L2=',L2,'FieldOffset=',FieldOffset,end=' ')

                if 'level' in rootgrp.variables:
                    #如果包含层，则使用层信息
                    Field[j,:,:]=np.mean(hgt[L1:L2,ilev,:,:],axis=0)
                else:
                    #如果不包含层，则不是使用层信息
                    Field[j,:,:]=np.mean(hgt[L1:L2,:,:],axis=0)

                print(np.shape( Field[j,:,:] ))
                j = j+1
                #hgt1 = hgt[L1:L2+1,int(SelLevel),:,:]
                #print( hgt[L1:L2,ilev,:,:])
                #print(hgt1.shape)
                #print(np.shape(,axis=0) ) )


    print('len_I_Year=%d,len lat= %d,len lon =%d'%(len(I_Year),len(lat),len(lon)) )
    print('Field Shape = ',Field.shape)
    rootgrp.close()
    return Field,dinfo

#------------------------------------------------------------------------------
def save_obj(object1,FileName='temp.npy',):
    '''
    cdiag1是对象 ,cdiag2是获取存储后的对象
    '''
    dt = np.dtype({'names': ['name1','cdiag1'],'formats': ['f','O']})
    a1 = np.array([(3,object1)],dtype=dt)
    f = file(FileName, "wb")

    np.save(f,a1)  #不压缩存储

    f.close()


###############################################################################

def load_obj(FileName='temp.npy'):
    f = file(FileName, "rb")
    a = np.load(f)
    #print('aaa=',a)
    object1 = a[0]['cdiag1']
    f.close()
    return object1

#------------------------------------------------------------------------------
#装饰器
def dectime2(str):
    mytic()
    def dectime(func):
        #*args表示元祖参数，**kargs表示字典参数
        def Function(*args,**kargs):
            return func(*args,**kargs)
        return Function
    mytoc(str)
    return dectime

#------------------------------------------------------------------------------
def repeat(n):
    def repeatn(f):
        def inner(*args, **kwds):
            for i in range(n):
                ret = f(*args, **kwds)
            return ret
        return inner
    return repeatn



#------------------------------------------------------------------------------
if __name__ == "__main__" :
    #打印结果
    print( cur_file_dir() )