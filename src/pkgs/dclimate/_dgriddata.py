# -*- coding: cp936 -*-
from __future__ import print_function
import sys,os,re,gc,time
import numpy as np
import time as getsystime
import timeit
import time
from mpl_toolkits.basemap import shapefile
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
def griddata_scipy_idw(x, y, z, xi, yi,function='linear'):
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
    interp = Rbf(x, y, z, function=function,epsilon=2)#linear
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

#-----------------克里金----------------------
#def griddata_kriging(x, y, z, xi, yi):
#def kriging(range,mode,Z_s,resol,pos,c0,c1,side_len):
def griddata_kriging(X,Y,Z,xi,yi,c0=1.5,c1=20, mode=2):
    '''''kriging------------------------------
       杜良敏从互联网的python代码中修改，具体的结果如下
    '''
    a_Range = xi.shape[0]+yi.shape[1]
    a_Range = a_Range/3
    #a_Range = 25

    #a_Range 为变程 Range
    #指区域化变量在空间上具有相关性的范围。在变程范围之内，数据具有相关性；而在变程之外，数据之间互不相关，即在变程以外的观测值不对估计结果产生影响。

    item = len(X)-1
    #file1=open("data.txt","w")
    #---------initialize values--------
    #----得到范围坐标-----

    # begin_row = range[0]
    # begin_col = range[1]
    # end_row = range[2]
    # end_col = range[3]

    dim = item +1
    #---分辨率-------

    value = np.ones((item+1,item+1))
    D = np.ones((item+1,1))
    Cd = np.zeros((item+1,item+1))

    i,j=0,0

    while i<item:
        j=i
        while j<item :
        #temp_i = pos[i]
            #temp_i_x = temp_i[0]
            #temp_i_y = temp_i[1]
            temp_i_x = X[i]
            temp_i_y = Y[i]

            #temp_j = pos[j]
            #temp_j_x = temp_j[0]
            #temp_j_y = temp_j[1]
            temp_j_x = X[j]
            temp_j_y = Y[j]

            test_t = (temp_i_x-temp_j_x)**2+(temp_i_y-temp_j_y)**2
            test_t = np.sqrt(test_t)
            #test_t = np.linalg.norm(np.array([temp_i_x-temp_j_x,temp_i_y-temp_j_y]))
            #np.linalg.norm(np.array([i_f-temp_k_x,j_f-temp_k_y]))

            Cd[i][j]= test_t #生成计算矩阵
            j=j+1
        i=i+1

    #print('Cd=',Cd.shape)

    #----------三种模型下变差函数的实现，放入v中----------------
    value[item][item]=0
    #print(value)
    #if 1==mode:
    #    value = np.where(Cd<a_Range,c0 + c1*(1.5*Cd/a_Range - 0.5*(Cd/a_Range)*(Cd/a_Range)*(Cd/a_Range)),c0+c1)

    i,j=0,0
    while i < item :
        j= i
        while j < item :
            if mode == 1 : #Spher mode
                if  Cd[i][j] < a_Range :
                    value[i][j] = value[j][i] = c0 + c1*(1.5*Cd[i][j]/a_Range - 0.5*(Cd[i][j]/a_Range)*(Cd[i][j]/a_Range)*(Cd[i][j]/a_Range))
                else:
                    value[i][j] = value[j][i] = c0 + c1
            if mode == 2: #  Expon mode
                value[i][j] = value[j][i] = c0 + c1*(1-np.exp(-3*Cd[i][j]/a_Range))
            if mode == 3:#Gauss mode
                pass
            j=j+1
        i=i+1
    #cnt_x = (end_row - begin_row)/resol_x#x方向步长
    #cnt_y = (end_col - begin_col)/resol_y#y方向步长
    #print cnt_x
    #print cnt_y
    print('value.shape=',value.shape)


    #sys.exit(0)
    #l = 0

    #print('resol_x=',resol_x)
    #print('resol_y=',resol_y)
    ###########################
    shape1 = xi.shape
    dat1 = np.zeros_like(xi)

    x2 = xi.flatten()
    y2 = yi.flatten()
    dat2 = dat1.flatten()

    for ii in xrange(len(x2)):

        #print(D)
        #print(D.shape)         print(value)
        #######################################

        #i_f=x2[ii]
        #j_f=y2[ii]

        temp_k_x = X#[k]
        temp_k_y = Y#[k]

        a1 = x2[ii] - X
        b1 = y2[ii] - Y
        test_t = np.sqrt(a1*a1+b1*b1)
        #test_t = np.linalg.norm(np.array([i_f -temp_k_x , j_f-temp_k_y] ))
        if(1==mode):
            D = np.where(test_t<a_Range,c0 + c1*(1.5*test_t/a_Range - 0.5*(test_t/a_Range)**3),c0+c1)
        if(2==mode):
            D = c0+ c1*(1-np.exp(-3*test_t/a_Range))
        if(3==mode):
            D = c0+ c1*(1-np.exp(-1*(3*test_t)**2/a_Range**2 ))
            #1-np.exp(-1*( (3*Cd[i][j])**2/a_Range*2 ) )
        D[-1]=1#并补一个1
        #print('111',test_t,test_t.shape,a1,b1,c1,D.shape)
        #sys.exit(0)

        ########################################
        #-----d v 均已求出，现在计算w
        #print(D)
        try :
            D = np.linalg.solve(value,D)
        except:
            print("Kinging linalg.solve error")

        #print(D)
        #sys.exit(0)
        #print(D.shape)
        test_t = np.sum(D*Z)

        dat2[ii]=test_t

    zi = dat2.reshape(shape1)
    return zi

#------------------------------------------------------------------------------
def griddata_linear_rbf_flatten(x, y, z, xi, yi):
    '''
    离散点插值为网格点的函数，速度较快
    网格点上限不要超过500x500会溢出
    '''
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)
    #(nx,ny)=xi.shape
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
    #zi = zi.reshape(nx,ny)
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

#------------------------------------------------------------------------------
def griddata_nearest(x, y, z, xi, yi):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

    (nx,ny)=xi.shape
    xi, yi = xi.flatten(), yi.flatten()
    from scipy.interpolate import griddata
    interp = griddata((x, y), z,(xi,yi), method='nearest')#linear
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

    if('kriging'==func):
        zi= griddata_kriging(x,y,z,xi,yi)


    if('scipy_idw'==func):
        zi= griddata_scipy_idw(x,y,z,xi,yi)

    if('line_rbf'==func):
        zi = griddata_linear_rbf(x,y,z,xi,yi)  #        grid3 = grid3.reshape((ny, nx))
        print(zi.shape,x.shape,y.shape,z.shape,xi.shape,yi.shape)
        #sys.exit(0)

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
    import hashlib
    m1=hashlib.md5(str(x1)+str(y1)+shapefilename)
    #print(str(x1)+str(y1)+shapefilename)
    md5filename = 'Z'+m1.hexdigest()+".npy"
    #print(md5filename)
    if(not os.path.isfile(md5filename)):
        #如果文件不存在，就重新计算mask矩阵
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
        #保存矩阵
        np.save(md5filename,grid1)
        #print(grid1)
    else:
        grid1 = np.load(md5filename)
        #print(tmp_ary1)


    #----------------------------------------------------------------------
    #sys.exit(0)
    return  grid1#,shapes
#------------------------------------------------------------------------------