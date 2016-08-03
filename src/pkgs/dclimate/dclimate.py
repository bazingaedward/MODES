#-*-coding:utf8-*-
from __future__ import division
from __future__ import print_function
#moduel1.py
import numpy as np
from numpy.linalg import *
from scipy.stats import *
import matplotlib.pyplot as plt
import copy
import math
import dfunc as df
import sys

version = '0.2'

######################################################
#获取中文字体
######################################################
def GetCnFont():
    import matplotlib,os
    windir = os.environ["windir"]
    font1 = os.path.join(windir,'Fonts','simsun.ttc')
    #cnfont = matplotlib.font_manager.FontProperties(fname='C:\WINDOWS\Fonts\simsun.ttc')
    cnfont = matplotlib.font_manager.FontProperties(fname=font1)
    return cnfont


def GetCnFont2(size1=8):
    import matplotlib,os
    windir = os.environ["windir"]
    font1 = os.path.join(windir,'Fonts','simsun.ttc')
    #cnfont = matplotlib.font_manager.FontProperties(fname='C:\WINDOWS\Fonts\simsun.ttc')
    cnfont = matplotlib.font_manager.FontProperties(fname=font1,size=size1)
    return cnfont

def GetCnFont2(size1=8):
    import matplotlib,os
    windir = os.environ["windir"]
    font1 = os.path.join(windir,'Fonts','simsun.ttc')
    #cnfont = matplotlib.font_manager.FontProperties(fname='C:\WINDOWS\Fonts\simsun.ttc')
    cnfont = matplotlib.font_manager.FontProperties(fname=font1,size=size1)
    return cnfont

######################################################
#任意组合工具
######################################################
def CombinationEnumerator(listObj, selection) :
    class CombEnum(object) :
        def __init__(self, listObj, selection) :
            assert selection <= len(listObj)
            self.items = listObj
            self.route = ([True, ]*selection) + [False, ]*(len(listObj)-selection)
            self.selection = selection
        def value(self) :
            result = [ val for flag, val in zip(self.route, self.items) if flag ]
            return result
        def next(self) :
            try :
                while self.route[-1] :
                    self.route.pop()
                while not self.route[-1] :
                    self.route.pop()
            except :
                return False

            self.route[-1] = False

            count = self.route.count(True)
            self.route.extend( (self.selection - count) * [True,] )
            padding = len(self.items) - len(self.route)
            self.route.extend( padding * [False, ])
            return True

    if selection == 0 :
        yield []
        return

    rotor = CombEnum(listObj, selection)
    yield rotor.value()
    while rotor.next() :
        yield rotor.value()


######################################################
#任意排列工具
######################################################
def PermutationEnumerator(choice, selection) :
    class Rotor(object) :
        def __init__(self, choice, selection, parent = None) :
            assert len(choice) >= selection
            self.selection = selection
            self.parent = parent
            self.choice = choice
            self.cursor  = 0
            if selection == 1 :
                self.child = None
            else :
                childChoice = choice[:self.cursor] + choice[self.cursor + 1:]
                self.child = Rotor(childChoice, selection - 1, self)
        def value(self) :
            if self.child :
                result = self.child.value()
                result.append(self.choice[self.cursor])
                return result
            else :
                return [ self.choice[self.cursor], ]
        def next(self) :
            node = self.child
            while node.child :
                node = node.child

            # descend to the lowest node
            node.cursor += 1
            if node.cursor < len(node.choice) :
                return True

            node = node.parent
            while len(node.choice) == node.cursor + 1 :
                node = node.parent
                if not node :
                    return False

            node.cursor += 1
            cursor = node.cursor
            node_child_choice = node.choice[:cursor] + node.choice[cursor + 1:]
            node.child = Rotor(node_child_choice, node.selection -1,  node)
            return True
    rotor = Rotor(choice, selection)
    yield rotor.value()
    while rotor.next() :
        yield rotor.value()

if __name__ == "__main__" :
    s = ['a', 'b', 'c', ]
    enum = PermutationEnumerator(s, 3)
    for i in enum :
        print(i)

    items = [ 1,2,3,4]
    enum = CombinationEnumerator(items, 2)
    for i in enum :
        print(i)
    print

    A = np.linalg.norm(-1*np.ones([4,4]))
    #RMSE = stats.linalg.norm(X-Y)
    print('A=',A)


def test1():
    print('aaaaaaaa\n')


###################################################
# dpcr 主成分回归
#和Matlab  差别很大，要注意
###################################################
def dpcr(X,Y):
    #print X
    #print 'Y=',Y
    U, S, V = np.linalg.svd(X)
    m,n=X.shape;
    S1=np.zeros((m,n),dtype=np.float)
    #print S1,np.size(S)

    for i in range(np.size(S)):
        S1[i,i]=S[i]

    #n = min([m,n])
    #    S1[:n,:n] = diag(S)

    #S=np.diag(S)
    #print 'S1=',S1
    #Pmat = Y*V*pinv(S)*U';
    Pmat =np.dot(Y, np.dot(V.T, np.dot(linalg.pinv(S1),U.T) ) )
    return Pmat


#################################################
def mapstd(b,PS=0,stat='run'):
    b = np.atleast_2d(b)
    if(stat=='run') :
        m,n=b.shape
        #print m,n
        bstd=b.std(axis=1,ddof=1)
        bmean=b.mean(axis=1)
        #print bstd
        bstd1=np.kron(np.ones((n,1),dtype=np.float),bstd).T
        #print('b=',b)
        #print('bstd=',bstd)
        #print('bstd1=',bstd1)
        bmean1=np.kron(np.ones((n,1),dtype=np.float),bmean).T
        #print bstd1.shape,bmean1.shape
        b=(b-bmean1)/bstd1
        PS={'std':bstd,'mean':bmean}
        #print 'bstd=',bstd
        #print a['A']#print a['B']
        return b,PS

    if(stat=='reverse'):
        m,n=b.shape
        #print m,n
        bstd = PS['std']
        bmean = PS['mean']
        bstd1=np.kron(np.ones((n,1),dtype=np.float),bstd).T
        bmean1=np.kron(np.ones((n,1),dtype=np.float),bmean).T
        #print bstd
        b=b*bstd1+bmean1
        return b

    if(stat=='apply'):
        m,n=b.shape
        #print m,n
        bstd = PS['std']
        bmean = PS['mean']
        bstd1=np.kron(np.ones((n,1),dtype=np.float),bstd).T
        bmean1=np.kron(np.ones((n,1),dtype=np.float),bmean).T
        #print bstd
        b=(b-bmean1)/bstd1
        return b



###################################################
#prestd
#中心化程序用于标准化
#b,bmean,bstd=prestd(b)
###################################################
def stdpre(b):
    b = np.atleast_2d(b)
    m,n=b.shape
    #print m,n
    bstd=b.std(axis=1)
    bmean=b.mean(axis=1)
    #print bstd
    bstd1=np.kron(np.ones((n,1),dtype=np.float),bstd).T
    bmean1=np.kron(np.ones((n,1),dtype=np.float),bmean).T
    b=(b-bmean1)/bstd1
    return b,bmean,bstd

###################################################
#poststd
#还原中心化程序
###################################################
def stdpost(b,bmean,bstd):
    m,n=b.shape
    #print m,n
    bstd1=np.kron(np.ones((n,1),dtype=np.float),bstd).T
    bmean1=np.kron(np.ones((n,1),dtype=np.float),bmean).T
    #print bstd
    b=b*bstd1+bmean1
    return b


###################################################
#transtd
#apply中心化程序
###################################################
def stdtran(b,bmean,bstd):
    m,n=b.shape
    #print m,n
    bstd1=np.kron(np.ones((n,1),dtype=np.float),bstd).T
    bmean1=np.kron(np.ones((n,1),dtype=np.float),bmean).T
    #print bstd
    b=(b-bmean1)/bstd1
    return b



###################################################
#EOF分解程序
###################################################
def deof(X):
    #print x
    m,n=X.shape

    if(m<=n):
        #print "m<=n"
        E,V=eig(np.dot(X,X.T)/float(n)) # V: eigenvectors; E: eigenvalues

        #E=np.flipud(np.fliplr(np.diag(E)))
        #lamda=E[::-1]
        I=np.argsort(E)
        I=I[::-1]
        lamda=E[I]
        V=V[:,I]
        T=np.dot(V.T,X);
        #V=np.fliplr(V);
        #T=np.flipud(T);
    else:
        #print "m>n"
        #print('a'*80)
        #print(X)
        #np.savetxt('X.txt',X)
        #print('b'*80)
        #print(np.dot(X.T,X))



        E,V=eig(np.dot(X.T,X))
        #V=np.fliplr(V[)
        V=np.dot(X,V)
        I=np.argsort(E)
        I=I[::-1]
        E = E[I];
        V=V[:,I]
        #print 'E=',E #np.sqrt(E)
        lamda=E/float(n)
        #print 'lamda=',lamda
        #V=linalg.solve(V, np.diag(np.sqrt(lamda)))
        #B/A is roughly the same as B*inv(A
        #V=np.dot(V, linalg.pinv(np.diag(np.sqrt(E)))  )
        #print(V,E)
        V=np.dot(V, np.diag(1.0 / np.sqrt(E))  )
        T=np.dot(V.T,X)
         #for ii=1:length(lamda)
        #    vr(1,ii)=(sum(lamda(1:ii))/sev)*100.0;
        #    vr(2,ii)=(lamda(ii)/sev)*100.0;
        #    vr(3,ii)=lamda(ii)*sqrt(2/n);
        #end

    #print np.size(lamda)
    #sev=lamda.sum();
    #vr=np.ones((3,np.size(lamda)),dtype=np.float)
    #for i in range(0,np.size(lamda)):
        #print 'i=',i
        #vr[0,i]=(lamda[0:i+1].sum()/sev)*100.0;
        #vr[1,i]=(lamda[i]/sev)*100.0;
        #vr[2,i]=lamda[i]*np.sqrt(2/n);
    V=np.where(np.isnan(V),0,V)
    T=np.where(np.isnan(T),0,T)
    lamda=np.where(np.isnan(lamda),0,lamda)
    return V,T,lamda

############################################
#EOF结果分析
############################################
def deofA(X):
    V,T,lamda=deof(X)
    n=X.shape[1]
    sev=lamda.sum();
    vr=np.ones((3,np.size(lamda)),dtype=np.float)
    for i in range(0,np.size(lamda)):
        #print 'i=',i
        vr[0,i]=(lamda[0:i+1].sum()/sev)*100.0;
        vr[1,i]=(lamda[i]/sev)*100.0;
        vr[2,i]=lamda[i]*np.sqrt(2.0/n);
    return V,T,lamda,vr

###################################################
#任意数矩阵乘方
###################################################
def mpower(A,P):
    '''Compute x raised to the power y when x is a square matrix and y
    is a scalar.'''

    s = np.shape(A)
    B=np.eye(s[0])
    if len(s) != 2 or s[0] != s[1]:
        raise ValueError('matrix must be square')
    if P == 0:
        return np.eye(s[0])
    if P <0:
        #print '----deci-------'
        A=np.linalg.inv(A)
        P=-1.0*P;
        #print 'inv A',A

    print(A)
    [D,V] = np.linalg.eig(A)         #eig 和 matlab比 输出值是反的，需要注意
    #D = np.diag(D)
    print('D=',D,'V=',V)
    print('D*',np.power(D,P))
    return( np.dot(V,np.dot( np.diag( D**P ) , np.linalg.inv(V) ) ) )
    #d*v.^ np.dot(y)*/d

#like matlab size
def mshape(X):
    if(X.ndim==1):
        return 1,X.shape[0]
    else:
        return X.shape


##################################################
#用于预先保存的数组
##################################################
def sim_like_array(x,y):
    x=x.ravel()
    y=y.ravel()
    E1 = np.in1d(x,y)
    F1 = np.where(E1,1,0)
    G1=np.sum(F1)/len(F1)
    return G1

#计算两个个场的数据是否相似
def sim_like_array2(x,y):
    x=x[:,0]
    y=y[:,0]
    E1 = np.in1d(x,y)
    F1 = np.where(E1,1,0)
    G1=np.sum(F1)/len(F1)
    return G1



def mcorr_pre_save(X,Y,fname = 'mcorr.dat'):
    import os
    if(os.path.exists(fname) ):
        #print(np.sum(D1)/len(D1))
        dict1 = df.load_obj(fname)
        X_Old = dict1['X']
        Y_Old = dict1['Y']
        SimX = sim_like_array2(X,X_Old)
        SimY = sim_like_array2(Y,Y_Old)

        print('SimX=%d SimY=%d'%(SimX,SimY))

        if(SimX>0.95 and \
           SimY>0.95 and \
           X.shape[0]==X_Old.shape[0] and\
           Y.shape[0]==Y_Old.shape[0]):
            r = dict1['r']
            p = dict1['p']
        else:
            dict1 = {}
            r,p = mcorr(X,Y)
            dict1['r']=r
            dict1['p']=p
            dict1['X']=X
            dict1['Y']=Y
            df.save_obj(dict1,fname)
    else:
        dict1 = {}
        r,p = mcorr(X,Y)
        dict1['r']=r
        dict1['p']=p
        dict1['X']=X
        dict1['Y']=Y
        df.save_obj(dict1,fname)

    return r,p


    #r,p = dclim.mcorr(Region_Obs2,Field3)


def presave_mcorr(Field,Obs,hdfile='corr_r_p.h5'):
    import hashlib,base64
    import pandas as pd
    str_hash_r = 'r_'+hashlib.md5(Field.tostring()+Obs.tostring()).hexdigest()
    str_hash_p = 'p_'+hashlib.md5(Field.tostring()+Obs.tostring()).hexdigest()

    store=pd.HDFStore(hdfile,'a',complib='blosc')
    #p4d=store['p4d']
    print(store)
    print(len(store))
    if(str_hash_r in store):
        r=store[str_hash_r].values
        p=store[str_hash_p].values
    else:
        r,p=mcorr(Field,Obs)
        df_r = pd.DataFrame(r)
        df_p = pd.DataFrame(p)
        store[str_hash_r]=df_r
        store[str_hash_p]=df_p
    #print(r)
    store.close()
    return r,p


###################################################
#对行向量球相关系数和P value
###################################################
def mcorr(X,Y):
    #X=ndim1to2(X)
    #Y=ndim1to2(Y)
    X=np.atleast_2d(X)
    Y=np.atleast_2d(Y)
    print(X.shape,Y.shape)
    m1,n1 = X.shape
    m2,n2 = Y.shape
    #print m1,n1
    #print m2,n2
    #对行求相关

    if n2 != n1:
        raise ValueError("lengths of x and y must match")
    r=np.zeros((m1,m2),dtype=np.float)
    p=np.zeros((m1,m2),dtype=np.float)

    #print r
    #print p
    import time
    starttime = time.clock()
    str2=''
    str3=''

    for i in range(m1):
        endtime = time.clock()
        res = endtime-starttime
        precent1 = float(i)*100/float(m1)
        if(0.0==precent1/100.0):
            FullTime=0
            EST=999
        else:
            FullTime= res/(precent1/100.0)
            EST=FullTime-res
        str2='%5.2f%%(%d) cost :%4.1f sec remain:%4.1f sec(Total Elapsed Time%5.1f)'%(precent1,m1,res,EST,FullTime)
        print(str2,end='')
        for j in range(m2):
            r[i,j],p[i,j] = stats.pearsonr(X[i,:],Y[j,:])
            #print stats.pearsonr(X[i,:],Y[j,:])
        print('\r',end='')
    print(str2)
    #print r
    #print p
    return r,p


###################################################
#类似于 atleast_2d函数
###################################################
def ndim1to2(X):
    if(X.ndim==1):
        #X.shape=1,X.shape[0]
        X.shape=1,-1
    return X



############################################################
#典型相关非常重要，写了两天两夜:-(
############################################################
def CCA(X,Y):
    #print 'Y=',Y
    #X=np.array([1,2]);Y=np.array([2,1])
    #print X.shape
    #X,t1,t2=dcl.prestd(X)
    #Y,t1,t2=dcl.prestd(Y)
    #print 'XS=', X.shape
    #print 'YS=',Y.shape
    Z=np.vstack((X,Y))
    #print 'Z=',Z
    C=np.cov(Z)
    #print 'C=',C

    sx = mshape(X)[0]
    sy = mshape(Y)[0]

    #print 'sx=',sx,'sy=',sy
    Cxx = C[0:sx, 0:sx]
    Cxy = C[0:sx, sx:sx+sy];
    Cyx = Cxy.T
    Cyy = C[sx:,sx:]

    #print 'Cxx=',Cxx
    #print 'Cxy=',Cxy
    #print 'Cyy=',Cyy

    invCyy = np.linalg.inv(Cyy);
    #print 'invCyy=',invCyy

    # --- Calcualte Wx and r ---

    #[Wx,r] = eig(inv(Cxx)*Cxy*invCyy*Cyx); % Basis in X
    #r = sqrt(real(r));      % Canonical correlations
    r,Wx = eig(np.dot(np.dot(np.linalg.inv(Cxx),Cxy),\
                      np.dot(invCyy,Cyx)) )
    r = np.sqrt(r)

    r = r[::-1]
    #print 'r=',r,'\nWx=',Wx
    V=np.fliplr(Wx);
    #print 'V=',V

    #[r,I]= sort((real(r)))
    I=np.argsort(r)
    r=r[I]
    r = r[::-1]

    #print 'r=',r
    #print 'I=',I;
    #print I.size

    #print 'Wx0=',Wx
    #for j in range(I.size):
    #    print j,I[j]
    #    Wx[:,j] = V[:,I[j]];  # sort reversed eigenvectors in ascending order

    Wx=V[:,I]

    #print 'Wx1=',Wx
    Wx = np.fliplr(Wx)
    #print 'Wx2=',Wx

    Wy = np.dot(np.dot(invCyy,Cyx),Wx)

    #Wy = np.array([[ 0.02411293  ,0.32115262, -0.16460067], \
    #    [ 0.0297546 , -0.19535438 , 0.51376637], \
    #    [ 0.02651068 ,-0.23710715, -0.15701734]])
    #print 'Wy1=',Wy

    #tmpw = np.linalg.matrix_power(,2);
    tmpw = np.abs(Wy)**2
    tmpw = np.sqrt( tmpw.sum(axis=0) )
    tmpw = np.kron(np.ones((sy,1),dtype=np.float),tmpw)
    Wy = Wy/tmpw
    del tmpw
    #print 'Wy2=',Wy

    Pmat=np.dot(np.dot(Cyy,Wy),np.diag(np.sqrt(r)))
    Pmat = np.dot(Pmat,Wx.T)
    #print 'r=',np.sqrt(r)
    #print 'Pmat=',Pmat #np.diag(np.sqrt(r))
    #Pmat=Cyy*Wy*diag(sqrt(r))*Wx';
    return Wx,Wy,r,Pmat


######################
#删除常量行
######################
def del_con_rows(X,PS=1,stat='run'):
    X=np.atleast_2d(X)
    if(stat=='run'):
        B=X.std(axis=1)!=0
        C=1-B;
        PS={'IndexTrue':B,'IndexFalse':C,'Con_Rows':X[C,:]}
        return X[B,:],PS

    if(stat=='reverse'):
        #bstd = PS['std']
        [m,n]=X.shape
        B=PS['IndexTrue']  #非常量行
        C=PS['IndexFalse']  #常量行
        XCon=PS['Con_Rows']
        m=B.shape[0]
        X2=np.zeros([m,n])
        #print 'rows=',m,'cols=',n
        X2[B,:]=X
        X2[C,:]=XCon
        #print X2
        return X2


################################################
#BP-CCA场对场预测
################################################
def dpre_eof_cca(Field,Region,Field_P,K):

    Field,PS_FM  = mapstd(np.hstack((Field,Field_P))  );

    Region,PS_RM = mapstd(Region);
    #print 'Field.shape=', Field.shape
    #np.linalg.svd(Field)
    #deof(Field)

    Field =  np.where(np.isnan(Field) ,np.random.random()*0.0000001,Field)
    Region = np.where(np.isnan(Region),np.random.random()*0.0000001,Region)

    W_F,H_F,D_F=deof(Field)
    W_R,H_R,D_R=deof(Region)

    H_F1,PS_F1M = mapstd(H_F[0:K,:])
    H_R1,PS_R1M = mapstd(H_R[0:K,:])

    Wx,Wy,r,Pmat=CCA(H_F1[:,0:-1],H_R1)

    H_R1Pre=np.dot(Pmat, H_F1[:,-1].T );

    H_R1Pre = ndim1to2(H_R1Pre);
    H_RPre= mapstd(H_R1Pre.T,PS_R1M,stat='reverse')

    X_Pre1 = np.dot(W_R[:,0:K],H_RPre)#[:,0])
    X_Pre= mapstd(X_Pre1,PS_RM,stat='reverse')
    return X_Pre

################################################
#EOF迭代场对场预测
################################################
def dpre_eof_ite(Field,Region,Field_P,K=4):
    K=K-1
    Field,PS_FM  = mapstd(np.hstack((Field,Field_P))  );
    Region,PS_RM = mapstd(Region);

    #AllPre=[]

    INPUT_Rows = Region.shape[0]

    #print(Region.shape)

    Region=np.hstack(  (Region,np.zeros((INPUT_Rows,1)) )   )

    #print(Region.shape)

    Field3 = np.vstack((Field,Region))
    Field_Tmp = Field3.copy()

    tmp2=Field_Tmp[-INPUT_Rows:,-1]
    #print(tmp2.shape)
    MinError=0.0001
    for ii in range(500):
        #print(Field_Tmp)

        #去除为NaN的值
        Field_Tmp = np.where(np.isnan(Field_Tmp),np.random.random()*0.001,Field_Tmp)
        np.savetxt('a.txt',Field_Tmp,fmt='%5.1f')

        W_F,H_F,_=deof(Field_Tmp)
        Field_Tmp=np.dot(W_F[:,0:K],H_F[0:K,:])
        tmp1 = Field_Tmp[-INPUT_Rows:,-1]

        #print('tmp1.shape=',tmp1.shape)
        Field_Tmp=Field3;
        Field_Tmp[-INPUT_Rows:,-1]=tmp1
        Error = np.linalg.norm(tmp2-tmp1)
        s1 = ('%3d,%8.5e,%8.3f')%(ii,Error,tmp1[0]);
        print(s1,end='')
        print('\r',end='')
        if (Error<MinError):
            break
        tmp2=tmp1;

    X_Pre= mapstd(np.atleast_2d(tmp1).T,PS_RM,stat='reverse')
    return X_Pre

################################################
#简单的多元线性回归
################################################
def mregress(X,Y):
    #linalg.lstsq(X,y)
    Y=np.atleast_2d(Y).T
    b=np.dot(np.linalg.pinv(X),Y)
    #r,p=mcorr(Y.T,np.dot(X,b).T)
    return b


def do_PS(X,Y,L1=20,L2=50):
    '''    ################################################
    #PS评分函数 结果包含PS,SC,ACC
    ################################################'''
    #print __name__
    #print __doc__
    #print X.ndim,X.size,X.shape
    #print Y.ndim,Y.size,Y.shape
    X=X.ravel()
    Y=Y.ravel()
    #print X.ndim,X.size,X.shape
    #print Y.ndim,Y.size,Y.shape
    m1=X.size
    m2=Y.size
    if(m1!=m2):
        #sys.stderr.write('data number must equal!')
        raise ValueError("Error:data number must equal!")
    else:
        N=m1

    X1=X*Y
    ##print('X=',X)
    ##print('Y=',Y)
    ##print('X1=',X1)
    ##X_NaN = np.where(X1 == np.nan,1,0)
    ##print(X_NaN)
    ##if(np.sum(X_NaN)>0):
    ##    sys.exit(0)

    X1_LT_0=np.where(X1>0,1,0)
    N0=np.sum(X1_LT_0)
    SC=float(N0)/float(N)

    #print X
    #print Y
    #print X1>0
    #print SC

    T1 = (X1<=0)  #获取不同号的数据值

    #print X[T1]
    #print Y[T1]
    #print abs(X[T1])<L1
    #print abs(Y[T1])<L1
    #print np.sum( (abs(Y[T1])<L1)*(abs(X[T1])<L1) )

    #获取不同号值小于L1的个数
    N0=N0+ np.sum( (abs(Y[T1])<L1)*(abs(X[T1])<L1) )

    #print 'N0=',N0

    #print X1

    ii=0
    for jj in range(np.size(X1)):
        if( (X1[jj]>0) and (abs(X[jj])>=L1) and ( abs(X[jj])< L2) \
                   and (abs(Y[jj])>=L1) and ( abs(Y[jj])< L2) ):
            ii=ii+1
        #print 'jj=',jj
    N1=ii;

    #print 'N1=',N1

    ii=0
    for jj in range(np.size(X1)):
        if( (X1[jj]>0) and (abs(X[jj])>=L2) and (abs(Y[jj])>=L2) ):
            ii=ii+1
        #print 'jj=',jj
    N2=ii;
    #print 'N2=',N2


    PS=100.0*(N0+0.5*N1+N2)/(N+0.5*N1+N2);
    #print 'PS=',PS
    ACC,tmp=stats.pearsonr(X,Y)
    #print ACC
    RMSE = np.linalg.norm(X-Y)
    return PS,SC,ACC,-RMSE



def SSA_Build_Matrix(x1,L):
    '''X,N,K=build_ssa_matrix(x1,L)'''
    N=x1.size;
    #print N

    if 0==L:
        raise ValueError("Error:L Can't Equal Zero(0)")

    if L>N/2.0:
        raise ValueError("Error:L lager then l!")
    K=N-L+1
    X=np.zeros((L,K))
    #print X.shape
    for i in range(K):
        #print i
        #print X[:,i].shape
        #print x1[i:L+i].shape
        X[:,i]=x1[i:L+i]
    #print X
    #np.savetxt('1.txt',X,'%7.2f');
    return X,N,K


def SSA_Rebuild(rca,L):
    rca = np.atleast_2d(rca)
    N=rca.shape[0]+rca.shape[1]-1
    K=N-L+1
    Lp = min(L,K)
    Kp = max(L,K)
    #print Lp,Kp
    y=np.zeros(N)
    #print y.shape
    #print 'Lp=',Lp
    for k in range(Lp-1):
        for m in range(k+1):
            #print '%d,%d'%(m,k-m)
            y[k]=y[k]+(1.0/(k+1))*rca[m,k-m]
        #print 'k=%02d y=%5.2f'%(k,y[k])
    #print '1'
    #-------------------------------------
    for k in range(Lp-1,Kp):
        for m in range(Lp):
            y[k]=y[k]+(1.0/(Lp))*rca[m,k-m]
        #print 'k=%02d y=%5.2f'%(k,y[k])
    #print '2'
    #print 'Kp=',Kp
    #print 'N=',N
    #-------------------------------------
    for k in range(Kp,N):
        #print '**%d,%d'%(k-Kp+1,N-Kp+1)
        for m in range(k-Kp+1,N-Kp+1):
            #print '--%d,%d'%(m,k-m)
            y[k]=y[k]+(1.0/(N-k))*rca[m,k-m]
        #if(k-Kp+1<N-Kp+1):
            #print 'k=%02d y=%5.2f'%(k,y[k])
    return y
    #end.....

##################################
#check 1-PS 2-SC 3-ACC
###################################
def PRE_AR(I_Year,x1,L1,L2,\
        scoreid=5,title='',outflag=False,\
        outimg=False,outtxt='arout.txt',showimg=False):
    #scoreid 为评分方法
    #1代表 PS 评分
    #2代表 SC 符一致率
    #3嗲表 ACC 距平相关系数
    #4代表 RMSE 均方差误差
    #5代表 CSC 双评分准则


    mylist = []
    #mylist.append(title+':\n');
    mylist.append('最优自回归模型:\n');
    A=np.array([])  #print A.shape

    if(np.max(np.abs(x1))>10.0):
        level=[-20,20]
    else:
        level=[-0.5,0.5]

    for L in range(1,int(x1.size/2.5-1)):
        X,N,K=SSA_Build_Matrix(x1[:-1],L)
        X=X.T
        #print X
        I_Year2=I_Year[L:]
        y=x1[L:]
        y=np.atleast_2d(y)
        #print X.shape,y.shape

        b=mregress(X,y)
        #print b
        Y2=np.dot(X,b)
        #print Y2
        PS,SC,ACC,RMSE=do_PS(Y2,y,L1,L2)
        CSC=CSC_Score(Y2,y,level,dbg=0)
        mylist.append( "阶数: %02d,PS评分 = %6.2f SC=%5.3f ACC=%5.2f RMSE=%f CSC=%f\n"%(L,PS,SC,ACC,RMSE,CSC) )
        B=np.array([L,PS,SC,ACC,RMSE,CSC])
        #B=np.atleast_2d(B)
        #print B
        A=np.append(A,B)

    #------------------------------------------------------
    #找出最好的阶数
    A=A.reshape(-1,6)
    print(A[:,scoreid])
    I=np.argmax(A[:,scoreid])
    L=int(A[I,0])
    mylist.append('最好的阶数为:%d \n'%(int(L)) )
    X,N,K=SSA_Build_Matrix(x1[:-1],L)
    I_Year2=I_Year[L:]
    y=x1[L:]
    y=np.atleast_2d(y)
    b=mregress(X.T,y)
    Y2=np.dot(X.T,b)
    PS,SC,ACC,RMSE=do_PS(Y2,y,L1,L2)
    tstr = "Ps=%5.2f,SC=%3.2f,Tcc=%3.2f,CSC=%5.2f"%(PS,SC,ACC,CSC)
    #------------------------------------------------------

    #------------------------------------------------------
    #做预报
    X,N,K=SSA_Build_Matrix(x1,L)
    I_Year3=np.append(I_Year2,I_Year2[-1]+1)
    Y3=np.dot(X.T,b).flatten()
    #print Y3.shape,I_Year2.shape,I_Year3[-1].shape
    spred_val =u'预测年: %d ,预报值 :%5.2f '%(I_Year3[-1],Y3[-1])
    print('\n\n 预测年: %d ,预报值 :%5.2f \n'%(I_Year3[-1],Y3[-1]) )
    #------------------------------------------------------
    #sys.exit(0)
    #是否输出结果，不提供参数不输出各类结论，用于预测
    if ( outimg ):
        #---------获取中文字体--------
        cnfont = GetCnFont()

        plt.clf()
        fig = plt.figure(1)
        ax=fig.add_subplot(111)
        ax.plot(I_Year2,y.flatten(),'c-*')
        #ax.hold(False)
        ax.hold(1)
        ax.plot(I_Year3,Y3.flatten(),'-s',color='#ee8d18')
        ax.plot(np.atleast_1d(I_Year3[-1]),np.atleast_1d(Y3[-1]),'-o',\
                ms=10, lw=2, alpha=1, mfc='red')

        mylist.append('自回归系数：\n')
        mylist.append(str(b.flatten()) )
        mylist.append('\n\n 预测年: %d ,预报值 :%5.2f \n'%(I_Year3[-1],Y3[-1]) )

        #ax.legend(('OBS', 'PRE_FIT','PREDICT'),0)
        #l = \
        ax.legend((u'观测', u'拟合',u'预测'),0,prop=cnfont)
        #l.get_title().set_fontproperties(cnfont)
        

        #title1 = title+' Opti AR Predciton'
        title1 = title+u'(最优自回归)'
        plt.title(title1,fontproperties=cnfont)

        ax.plot(I_Year3,np.zeros_like(I_Year3),'r-')

        #ax.plot(I_Year3,np.ones_like(I_Year3)*L1,'g-')
        #ax.plot(I_Year3,np.ones_like(I_Year3)*L1*-1,'g-')
        #ax.plot(I_Year3,np.ones_like(I_Year3)*L2,'b-')
        #ax.plot(I_Year3,np.ones_like(I_Year3)*L2*-1,'b-')
        #plt.xlabel(u'time')
        plt.xlabel(u'时间',fontproperties=cnfont)
        plt.text(I_Year2.min(),y.min() ,spred_val,fontproperties=cnfont)

        #画预测值，或评分值
        #plt.text(I_Year2.min(),y.min() ,tstr)
        #plt.text(I_Year2.min(),0 ,tstr)

        ax.grid(True)
        plt.savefig(outimg)
        #print dir(plt)
        myfile = open(outtxt, 'w')
        myfile.writelines(mylist)
        myfile.close()

        if showimg:  #是否显示图像
            plt.show()

    print('\n\n 预测年: %d ,预报值 :%5.2f \n'%(I_Year3[-1],Y3[-1]) )
    print(I_Year3,Y3)
    print(I_Year3.shape,Y3.shape)

    return Y3,I_Year3
    #end of AR


########################################
#最优子集回归
########################################
def PRE_OSR(x1,idx2,idx_forPre1,L1,L2,\
        I_Year,Head_idx,\
        Select_Count =10,Check_Years=15,title='',\
        outflag=False,scoreid=5,\
        outimg='osrpre.png',outtxt='osrout.txt',showimg=False):
    #------------------------------------------------
    idx3,I2 = del_con_rows(idx2.T)
    I2=I2['IndexTrue']
    Head_idx2  = Head_idx[I2]
    idx_forPre3 = idx_forPre1[I2]
    print('Idx_forPre3.shape=',idx_forPre3.shape)

    #------------------------------------------------
    if(max(np.abs(x1))>10.0):
        level = [-30,30]
    else:
        level = [-1,1]

    print('level=',level)
    r,p=mcorr(x1,idx3) #Region)#idx3.T)
    #r=np.atleast_1d(r)
    print(r.shape,p.shape,idx2.shape,x1.shape)

    #排序
    I=np.argsort(-np.abs(r.flatten()))

    #获取指数头信息
    Head_idxS = Head_idx2[I]
    Head_idxS2 = Head_idxS[:Select_Count]

    #给预报用的量
    idx_forPre3 = idx_forPre3[I]

    #给训练用的量
    idx_S = idx3[I,:]
    idx_Select = idx_S[:Select_Count,:]

    #给预报用的量
    idx_forPre_Select = idx_forPre3[:Select_Count]

    print('idx_forPre_Select.shape=',idx_forPre_Select.shape)

    #print idx_S.shape,idx_S2.shape
    #print Head_idxS2
    #print idx_S2.T
    print(r[:,I[:Select_Count]])


    A=np.arange(Select_Count)
    item=A.tolist()
    del A

    mylist = list([])
    mylist.append('相关性最好的指数\n')
    mylist.append(str(Head_idxS2.tolist() )+'\n' )
    mylist.append('相关系数\n')
    #mylist.append(str(r[:,I[:Select_Count]].tolist() )+'\n' )
    mylist.append(np.array_str(r[:,I[:Select_Count]] )+'\n' )


    Combins=list([])
    #生成任意的排列组合
    for i in range(2,Select_Count):
        #print i
        enum = CombinationEnumerator(item, i)
        for Ct in enum :
            Combins.append(Ct)

    print(len(Combins))

    idx_Select=idx_Select.T

    L=0
    y=x1
    ChooseList=list([])
    for Ct in Combins:
        Its =  np.array(Ct)
        #print Its
        #b=regress(Ys,X2)idx_Select

        X=idx_Select[:,Its]
        #b=dclm.mregress(X,y)
        b=mregress(np.hstack((X, np.ones((X.shape[0],1)))),y)
        Y2=np.dot(np.hstack((X,np.ones((X.shape[0],1)))),b).flatten()
        PS,SC,ACC,RMSE=do_PS(Y2[-Check_Years:],y[-Check_Years:],L1,L2)
        CSC=CSC_Score(Y2[-Check_Years:],y[-Check_Years:],level)
        #mylist.append( "阶数: %02d,PS评分 = %6.2f 符号一致率=%5.3f TCC=%5.3f\n"\
        #%(L,PS,SC,ACC) )
        ChooseList.append([L,PS,SC,ACC,RMSE,CSC])
        L+=1
        #print Ct,
        #print '\r',
        #print b.shape

    ############挑最大值################
    ChooseList = np.array(ChooseList)
    #print(ChooseList)
    #print('scoreid=',scoreid)
    IMAX=np.argmax(ChooseList[:,scoreid])
    #print(ChooseList[IMAX,:])
    #sys.exit(0)
    L=int(ChooseList[IMAX,0])
    Ct=np.array(Combins[L])
    #print Ct
    Its =  np.array(Ct)
    print(Its)
    #sys.exit(0)
    X=idx_Select[:,Its]
    print('X.shape=',X.shape)
    b=mregress(np.hstack((X, np.ones((X.shape[0],1)))),y)
    Y2=np.dot(np.hstack((X, np.ones((X.shape[0],1)))),b).flatten()

    #PS,SC,ACC=dclm.do_PS(Y2,y,L1,L2)
    PS,SC,ACC,RMSE=do_PS(Y2[-Check_Years:],y[-Check_Years:],L1,L2)
    print("K=: %02d,PS = %6.2f SC =%5.3f TCC=%5.3f CSC=%f\n"%(L,PS,SC,ACC,CSC) )
    mylist.append("阶数: %02d,PS评分 = %6.2f 符号一致率=%5.3f TCC=%5.3f,RMSE=%6.3f\n"%\
        (L,PS,SC,ACC,RMSE) )


    X=idx_forPre_Select[Its]
    print( 'X=',np.append(X,1) )
    Y3 = np.dot(np.append(X,1),b)
    print( 'Y3=',Y3 )
    mylist.append('系数\n')
    mylist.append( str(b.tolist() ) )
    mylist.append("\n预测值%6.2f"%(Y3))
    tstr = "Last %d Years Ps=%5.2f,SC=%3.2f,Tcc=%3.2f,Pre=%5.2f"%(Check_Years,PS,SC,ACC,Y3)
    pstr = u"预测年: %d ,预报值 :%5.2f"%(I_Year[-1]+1,Y3[-1])


    #---------------------------------------------------
    if ( outflag or showimg ):
        #-------------------get font------------------------

        cnfont = GetCnFont()
        ########
        plt.clf()
        fig = plt.figure(1)
        ax=fig.add_subplot(111)
        #ax.plot(abs( r[:,I[:10]].flatten()))
        #画实况
        ax.plot(I_Year,y.flatten(),'-*')
        plt.text(I_Year.min(),y.min() ,pstr,fontproperties=cnfont)


        #ax.plot(I_Year,Y2.flatten())

        I_Year2 = np.append(I_Year,I_Year[-1]+1)
        Y2 = np.append(Y2.flatten(),Y3)

        #画回报检验
        ax.plot(I_Year2,Y2.flatten(),'-s')

        #画预测值
        ax.plot(np.atleast_1d(I_Year2[-1]),np.atleast_1d(Y3[-1]),'-^',\
                     ms=10, lw=2, alpha=1, mfc='red')

        #ax.legend(('OBS', 'PRE_FIT','PREDICT'),0)
        ax.legend((u'观测', u'拟合',u'预测'),0,prop=cnfont)
        ax.plot(I_Year2,np.zeros_like(I_Year2),'r-')
        #title1 = title+u' Opti Subset Rregress Predciton'
        title1 = title+u'预测'
        plt.title(title1,fontproperties=cnfont)

        plt.grid(True)
        plt.savefig(outimg)
        myfile = open(outtxt, 'a')
        myfile.writelines(mylist)
        myfile.close()

        if showimg:  #是否显示图像
            plt.show()              #print r[:,I.flatten()]
    #Y3预报值,Y2历史回报检验
    #return I_Year2,Y3,Y2
    return Y3,Y2



############################################################
# 获取最大的旋转分量 varimax
# REOF 杜良敏自己编写
# 2014-04-29
############################################################
def d_varimax(A,ite_eps=1e-5,ite_num=1000):
    pass
    M,N= A.shape
    #print(M,N)
    #print(A)
    #print(A*A)
    #return
    hj=np.sqrt(np.sum(A*A,axis=1))
    print(hj)
    print('-------- Varimax Rotation by DuLiangMin ------------')
    VV=sum(np.std(A,axis=0,ddof=1))
    print('VV=',VV)
    for L in range(ite_num):
        #print(L)
        for l in range(N-1):
            #print('-',l)
            for k in range(l+1,N):
                lj=A[:,l]/hj   # notation here closely follows Harman
                kj=A[:,k]/hj
                uj=lj*lj-kj*kj
                vj=2*lj*kj
                D=2*np.sum(uj*vj)
                H=np.sum(uj)
                E=np.sum(vj)
                C= np.sum(uj*uj-vj*vj)

                EE=D-2*H*E/M
                FF=C-(H*H-E*E)/M
                #import math
                phi=np.arctan2(EE,FF)
                phi=phi/4
                TA=np.eye(N)
                TA[l,l]=np.cos(phi)
                TA[k,k]=np.cos(phi)
                TA[l,k]=-np.sin(phi)
                TA[k,l]=np.sin(phi)
                A=np.dot(A,TA)

                #print(C,EE,FF,phi)
                #print(A)
                #return
            #---------------------
        ccc= np.abs(np.sum(np.std(A,axis=0,ddof=1))-VV)
        print('Item Times =',L,'Ite eps =',ccc,end='  ')
        #fprintf('%d,%f\n',L,ccc);
        if(ccc< ite_eps):
            print('')
            break
        VV=np.sum(np.std(A,axis=0,ddof=1))
        print(VV)
        #return
    V2=A;
    print('-------------------------------------------');
    #print(V2)
    return V2


############################################################
# 获取最大的旋转分量 varimax
# REOF
############################################################     
def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from scipy import eye, asarray, dot, sum
    from scipy.linalg import svd
    from numpy import diag
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)


def Level_Obs(Obs,level,dbg=False):
    Obs1=copy.deepcopy(Obs)
    Level_Obs2=copy.deepcopy(Obs)
    level=np.hstack((-np.Inf,level,np.Inf))
    if(dbg): print(level,len(level))
    #for ii=1:length(level)-1
    for ii in range(len(level)-1):
        a1 = level[ii]
        a2 = level[ii+1]
        if(dbg): print(ii,a1,a2)
        #b1 = (Obs1>a1)&(Obs1<=a2);
        b1 = np.logical_and(Obs1>a1,Obs1<=a2)
        Level_Obs2[b1]=ii

    return Level_Obs2

############################################################    
#双评分准则子程序
def CSC_S2(Obs,Pre,level,dbg=False):
    Obs_Lev=Level_Obs(Obs,level)
    Pre_Lev=Level_Obs(Pre,level)
    if(dbg): print('CSC_S2 Obs_Lev=',Obs_Lev+1)
    if(dbg): print('CSC_S2 Pre_Lev=',Pre_Lev+1)

    len2 = len(level)+1;
    if(dbg): print(len2)

    N=np.zeros((len2,len2))
    if(dbg): print(N)
    for ii in range(len2):
        for jj in range(len2):
            #print(ii,jj)
            N[ii,jj]= np.sum( Pre_Lev[Obs_Lev==ii]==jj )

    if(dbg): print(N)
    Nj = np.sum(N,axis=0)
    Ni = np.sum(N,axis=1)
    n=np.sum(N)
    if(dbg): print(Nj,Ni,n)

    a =np.nan_to_num( np.log(N) )*N
    a=np.nansum(a)
    if(dbg): print(a)

    sum_Ni = np.nansum( Ni* np.nan_to_num( np.log(Ni) ) )
    sum_Nj = np.nansum( Nj* np.nan_to_num( np.log(Nj) ) )

    S2 = 2*(a+n*math.log(n)-sum_Ni-sum_Nj)
    if(dbg): print(u'S2=',S2)
    return S2
    
###############################################
#双评分准则
def CSC_Score(Obs,Pre,level,L=1,dbg=False):
    '''
     L 因子个数
    '''
    Obs = Obs.flatten(1)

    #print('PRE=-------------------------------------------')
    #print(Pre)
    #print('PRE=-------------------------------------------')

    #sys.exit(0)
    Pre = Pre.flatten(1)

    #,Obs_avg=0
    Obs_avg = np.mean(Obs)
    #####################
    S2 = CSC_S2(Obs,Pre,level,dbg=dbg)
    #####################
    #Obs_avg =np.mean(Obs)
    Obs_Len = len(Obs)
    Qx = sum((Obs-Obs_avg)**2)/Obs_Len
    Qk = sum((Obs-Pre)*(Obs-Pre))/Obs_Len
    if(dbg):print('Qx=%f,Qk=%f,Qk/Qx=%f'%(Qx,Qk,Qk/Qx))

    S1=(len(Obs)-L)*(1-Qk/Qx)

    #if(dbg):print('S1=%f'%S1),
    #if(dbg):print('S2=%f'%S2)
    if(dbg):print('S1=%f,S2=%f,S1+S2=%f'%(S1,S2,S1+S2))
    return S1+S2;



##############################################
#气候特征相似预测，得出合成的结论，并返回相似年
# 2013-07-24
#__author_dlm__
#f010
##############################################
def dpre_climsim_ens(Field,Region,I_Year_Obs,Field_P,Sim_Year_Count = 8):

    #print(' Start Climate Sim ens preding.... f010')

    #print('Field.shape=',Field.shape)
    #print('Region.shape=',Region.shape)
    #print('Field_P.shape1=',Field_P.shape)

    Field_P = np.atleast_2d(Field_P).T

    #print('Field_P.shape2=',Field_P.shape)

    from scipy.spatial import distance
    Y = distance.cdist(np.atleast_2d(Field[:,0]),np.atleast_2d(Field[:,1]), 'euclidean')

    #print(Y,Y.shape)

    dis1 = np.zeros(Field.shape[1])

    #print('dis1.shape=',dis1.shape)

    for ii in range(Field.shape[1]):
        #print(ii)
        Y = distance.cdist(np.atleast_2d(Field[:,ii]),Field_P.T, 'euclidean')
        #print(Y,Y.shape)
        #print(Y[0,0])
        dis1[ii]=Y[0,0]

    #print(dis1)

    dis1_arg_sort = np.argsort(dis1)

    #print(dis1_arg_sort)
    #print(dis1[dis1_arg_sort])
    #print(I_Year_Obs[dis1_arg_sort])

    #选择相似的年份，作合成分析
    I_Year_Sort  = I_Year_Obs[dis1_arg_sort]
    I_Year_Sim  =I_Year_Sort[0:Sim_Year_Count+1]

    #对预报对象场排序
    Region_Sort  = Region[:,dis1_arg_sort]

    #print(Region_Sort)
    #选择最相似的8年
    Region_Sort_Select = Region_Sort[:,0:Sim_Year_Count+1]

    Region_Pred = np.mean(Region_Sort_Select,axis=1)
    #print('Region_Pred=',Region_Pred)
    #print(Region.shape,Region_Pred.shape)

    #print(' End Climate Sim ens preding....   f010')
    return Region_Pred,I_Year_Sim

##############################################
#利用因子过滤法对CCA方法的改进
# 2013-07-25
#__author_dlm__
#f010
##############################################
def CCA_Imporve1(Field3,Region_Obs2,Field_P):
    print(Field_P.shape)


    #r,p = dclim.mcorr(Region_Obs2,Field3)
    r,p = mcorr_pre_save(Region_Obs2,Field3)
    print(r.shape)
    print(p.shape)


    #def Filter_Pred_from_corr(self,Threshold=0.10):
    '''
        检验预报场和实况场的相关
    '''
    Threshold = 0.01
    p2 = np.where(p<Threshold,1,0)
    p_Field_sel = np.where(np.sum(p2,axis=0)>=2,True,False)

    print('p_Field_sel.shape=',p_Field_sel.shape,  np.sum( np.where(p_Field_sel,1,0) ))

    #p_obs_sel =  np.where(np.sum(p2,axis=1)>=2,True,False)
    #print('p_obs_sel.shape=',p_obs_sel.shape,  np.sum( np.where(p_obs_sel,1,0) ))

    #p_np2 = np.where(self.p_np<Threshold,True,False)

    print(Field3.shape)
    Field_Filter = Field3[p_Field_sel,:]
    Field_P_Filter = Field_P[p_Field_sel,:]
    #print(Field)
    X_Pre=dpre_eof_cca(Field_Filter,Region_Obs2,Field_P_Filter,4)
    #X_Pre=dpre_eof_ite(Field_Filter,Region_Obs2,Field_P_Filter,4)

    #print(X_Pre)
    return X_Pre.real


############################################################
# 获取站点的信息
############################################################
def Read_Region_Dat(FileName):
    RegionR = np.loadtxt(FileName)
    Region_LeftHead = RegionR[:,0:3]
    Region = RegionR[1:,3:]
    I_Year_Region = RegionR[0,3:]
    StaLatLon = RegionR[1:,0:3]
    print(Region.dtype)
    dict_Region={}
    dict_Region['RegionR']=RegionR
    dict_Region['LeftHead']=Region_LeftHead
    dict_Region['I_Year']=I_Year_Region
    dict_Region['LatLon']=StaLatLon
    dict_Region['Region']=Region
    return dict_Region



#######################################################################
#最优子集预测v3版本
#2014年2月17日加入支持向量机算法
#######################################################################
def optimal_subset_regression_v4(factor_h,obs1,factor_p,vali_obj=0,factors_count = 8):
    '''
    #vali_obj为验证方式
    #0-距离平符号一致率
    #1-ACC
    #2-最小均方差误差
    #3-自定义双评分 SC+ACC
    #4-双评分准则
    #from sklearn.svm import SVR
    #还有改进的余地，当输入的因子和预报对象一致时，自动调用预先存储的md5文件信息即可
    '''
    from sklearn.linear_model import BayesianRidge, LinearRegression
    '''

      最优子集回归第二版本
      Vali_obj=0 距平符号一致率来选择
      Vali_obj=1 最大相关系数率来选择
      Vali_obj=4 4双评分准则
    '''
    ## print('---OSR v2-- | vali_obj= %d '%(vali_obj),end='')

    #删除常量行,指数因子的循环非常重要
    #print('factor_h=',factor_h)
    #print('factor_p=',factor_p)
    # f1,i1 = dclimate.del_con_rows(factor_h)
    f1,i1 = del_con_rows(factor_h)
    #print(f1)
    #print(i1['IndexTrue'])

    factor_h=factor_h[i1['IndexTrue'],:]
    factor_p=factor_p[i1['IndexTrue'],:]

    #print('factor_h=',factor_h)
    #print('factor_p=',factor_p)
    #sys.exit(0)

    # factor_h,PS_h = dclimate.mapstd(factor_h)
    factor_h,PS_h = mapstd(factor_h)
    factor_p = mapstd(factor_p,PS_h,stat='apply')
    obs_Std,obs_PS_Std = mapstd(obs1)
    obs_Std = np.ravel(obs_Std)
    ## print('h,p,obs shapes =',factor_h.shape,factor_p.shape,obs1.shape,end='')

    #sys.exit(0)
    #-----------------获取所有的因子排列组合--------------

    if(factor_h.shape[0]<factors_count):
        factors_count=factor_h.shape[0]

    factor_combins=Build_Factor_Combins(factors_count)
    #print(factor_combins)
    print(u'组合个数=',len(factor_combins),end='')
    #-----------------获取所有的因子排列组合--------------
    #-----------------获取所有的因子排列组合--------------
    Choose_Factor = np.zeros((len(factor_combins)))

    #Regression = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #Regression = SVR(kernel='linear', C=1e3)
    #Regression = SVR(kernel='poly', C=1e3, degree=2)
    Regression = LinearRegression()
    #Regression = BayesianRidge()#compute_score=True)

    for i in range(len(factor_combins)):
        #if(i>1):
        #    print('%d '%i,end='')
        #    print('\r',end='')

        #print(factor1.shape,obs1.shape)
        sel1 = factor_combins[i]
        sel1 = np.array(sel1)
        factor2 = factor_h[sel1,:]
        #print(i,factor2)
        #Coef,residuals,rank,s  = np.linalg.lstsq(factor2.T,obs1.T)
        Regression.fit(factor2.T,obs_Std)

        #会算的拟合
        #Fitting = np.dot(factor2.T,Coef)
        Fitting = Regression.predict(factor2.T)

        # obs_hindcast = dclimate.mapstd(Fitting,obs_PS_Std,stat='reverse')
        obs_hindcast = mapstd(Fitting,obs_PS_Std,stat='reverse')

        Fitting = np.ravel(obs_hindcast)

        # if(i>61):
        #     print(factor2.T)
        #     print(obs_h.T)
        #     print(Fitting)
        #     print(obs1)
        #     print(obs_hindcast)
        #     sys.exit(0)


        #ACC,tmp=stats.pearsonr(X,Y)
        #print ACC
        #print(Fitting.shape,obs1.shape)
        #RMSE = np.linalg.norm(Fitting-obs1)
        #SC=dclimate.CSC_Score(Fitting,obs1,[-20,20])
        # PS,SC,ACC,RMSE=dclimate.do_PS(Fitting,obs1,20,50)
        PS,SC,ACC,RMSE=do_PS(Fitting,obs1,20,50)

        #RMSE=-ACC
        #RMSE=-SC
        #print('RMSE=',RMSE)
        #0距平相关系数 1 ACC 2 最小均方差误差 3自定义双评分 SC+ACC 4双评分准则
        if(0==vali_obj):
            Choose_Factor[i]=-SC

        if(1==vali_obj):
            Choose_Factor[i]=-ACC

        if(2==vali_obj):
            Choose_Factor[i]=-RMSE

        if(3==vali_obj):
            Choose_Factor[i]=-SC-ACC
            #print(u'选择因子个数=',len(sel1))
        if(4==vali_obj):
            CSC=CSC_Score(Fitting,obs1,[-20,20],len(sel1),dbg=False)
            Choose_Factor[i]=-CSC
            #np.dot(factor1.T,Coef)
        #print(Choose_Factor)
    print('\n',end='')
    sort1 = np.argsort(Choose_Factor)
    #print(Choose_Factor[sort1,:])
    #print(sort1)
    #print(factor_combins[sort1[0]])

    sel1 = factor_combins[sort1[0]]
    sel1 = np.array(sel1)
    print(u'选中对象个数=',sel1.shape,end='')
    factor2 = factor_h[sel1,:]

    #sys.exit(0)
    #print(i,factor2)

    #Coef,residuals,rank,s  = np.linalg.lstsq(factor2.T,obs1.T)
    #Fitting = np.dot(factor2.T,Coef)

    Regression.fit(factor2.T,obs_Std)
    Fitting = Regression.predict(factor2.T)
    obs_hindcast = mapstd(Fitting,obs_PS_Std,stat='reverse')
    Fitting = np.ravel(obs_hindcast)


    PS,SC,ACC,RMSE=do_PS(Fitting,obs1,20,50)
    #factor3 = dclimate.mapstd(factor0,PS0,stat='apply')
    ## print(u'|因子位置=',sel1,factor_p.shape,end='')
    #sys.exit(0)
    factor3 = factor_p[sel1,:]

    #Pred = np.dot(factor3.T,Coef)

    Pred = Regression.predict(factor3.T)
    print(u'|因子系数=',Regression.coef_,end='')
    print(u'|所选因子序列为',sel1)
    #sys.exit(0)

    Pred = mapstd(Pred,obs_PS_Std,stat='reverse')
    Pred = np.ravel(Pred)
    print(Pred)
    #sys.exit(0)
    ## print(u'预报值 = ',Pred,Pred.shape)
    return Pred

##########################################################
#生成因子序列提供给最优子集使用
###########################################################
def Build_Factor_Combins(iCount):
    item = []
    for i in range(iCount):
        item.append(i)

    factor_combins=[]
    for i in range(1,iCount+1):
        #print('___________%d_____________'%i)
        #enum = dclimate.CombinationEnumerator(item, i)
        enum = CombinationEnumerator(item, i)
        #print(enum)

        for Ct in enum:
            #print(np.array(j))
            factor_combins.append(Ct)
            #print(Ct)
    return factor_combins



class optimal_subset_regression_class1():
    """
    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.
    """

    def __init__(self):
        print("initialize... OSR ")

    def fit(self, factor_h,obs1,vali_obj=0, factors_count = 8):
        '''
        #vali_obj为验证方式
        #0-距离平符号一致率
        #1-ACC
        #2-最小均方差误差
        #3-自定义双评分 SC+ACC
        #4-双评分准则
        #from sklearn.svm import SVR
        #还有改进的余地，当输入的因子和预报对象一致时，自动调用预先存储的md5文件信息即可
        '''
        from sklearn.linear_model import BayesianRidge, LinearRegression
        '''
          最优子集回归第三版本
          Vali_obj=0 距平符号一致率来选择
          Vali_obj=1 最大相关系数率来选择
          Vali_obj=4 4双评分准则
        '''
        ## print('---OSR v2-- | vali_obj= %d '%(vali_obj),end='')

        #删除常量行,指数因子的循环非常重要
        #print('factor_h=',factor_h)
        #print('factor_p=',factor_p)
        #f1,i1 = dclimate.del_con_rows(factor_h)
        #f1,i1 = dclim.del_con_rows(factor_h)

        #print(f1)
        #print(i1['IndexTrue'])

        #factor_h=factor_h[i1['IndexTrue'],:]
        #print('factor_h=',factor_h)
        #print('factor_p=',factor_p)
        #sys.exit(0)

        # factor_h,PS_h = dclimate.mapstd(factor_h)
        factor_h,PS_h = mapstd(factor_h)

        obs_Std,obs_PS_Std = mapstd(obs1)
        obs_Std = np.ravel(obs_Std)
        ## print('h,p,obs shapes =',factor_h.shape,factor_p.shape,obs1.shape,end='')

        #sys.exit(0)
        #-----------------获取所有的因子排列组合--------------

        if(factor_h.shape[0]<factors_count):
            factors_count=factor_h.shape[0]

        factor_combins=Build_Factor_Combins(factors_count)
        #print(factor_combins)
        print(u'组合个数=',len(factor_combins),end='')
        #-----------------获取所有的因子排列组合--------------
        #-----------------获取所有的因子排列组合--------------
        Choose_Factor = np.zeros((len(factor_combins)))

        #Regression = SVR(kernel='rbf', C=1e3, gamma=0.1)
        #Regression = SVR(kernel='linear', C=1e3)
        #Regression = SVR(kernel='poly', C=1e3, degree=2)
        Regression = LinearRegression()
        #Regression = BayesianRidge()#compute_score=True)

        for i in range(len(factor_combins)):
            #if(i>1):
            #    print('%d '%i,end='')
            #    print('\r',end='')

            #print(factor1.shape,obs1.shape)
            sel1 = factor_combins[i]
            sel1 = np.array(sel1)
            factor2 = factor_h[sel1,:]
            #print(i,factor2)
            #Coef,residuals,rank,s  = np.linalg.lstsq(factor2.T,obs1.T)
            Regression.fit(factor2.T,obs_Std)

            #会算的拟合
            #Fitting = np.dot(factor2.T,Coef)
            Fitting = Regression.predict(factor2.T)

            # obs_hindcast = dclimate.mapstd(Fitting,obs_PS_Std,stat='reverse')
            obs_hindcast = mapstd(Fitting,obs_PS_Std,stat='reverse')

            Fitting = np.ravel(obs_hindcast)

            # if(i>61):
            #     print(factor2.T)
            #     print(obs_h.T)
            #     print(Fitting)
            #     print(obs1)
            #     print(obs_hindcast)
            #     sys.exit(0)


            #ACC,tmp=stats.pearsonr(X,Y)
            #print ACC
            #print(Fitting.shape,obs1.shape)
            #RMSE = np.linalg.norm(Fitting-obs1)
            #SC=dclimate.CSC_Score(Fitting,obs1,[-20,20])
            # PS,SC,ACC,RMSE=dclimate.do_PS(Fitting,obs1,20,50)
            PS,SC,ACC,RMSE=do_PS(Fitting,obs1,20,50)

            #RMSE=-ACC
            #RMSE=-SC
            #print('RMSE=',RMSE)
            #0距平相关系数 1 ACC 2 最小均方差误差 3自定义双评分 SC+ACC 4双评分准则
            if(0==vali_obj):
                Choose_Factor[i]=-SC

            if(1==vali_obj):
                Choose_Factor[i]=-ACC

            if(2==vali_obj):
                Choose_Factor[i]=-RMSE

            if(3==vali_obj):
                Choose_Factor[i]=-SC-ACC
                #print(u'选择因子个数=',len(sel1))
            if(4==vali_obj):
                CSC=CSC_Score(Fitting,obs1,[-20,20],len(sel1),dbg=False)
                Choose_Factor[i]=-CSC
                #np.dot(factor1.T,Coef)
            #print(Choose_Factor)
        print('\n',end='')
        sort1 = np.argsort(Choose_Factor)
        #print(Choose_Factor[sort1,:])
        #print(sort1)
        #print(factor_combins[sort1[0]])

        sel1 = factor_combins[sort1[0]]
        sel1 = np.array(sel1)
        print(u'选中对象个数=',sel1.shape,end='')
        factor2 = factor_h[sel1,:]
        #sys.exit(0)
        #print(i,factor2)
        #Coef,residuals,rank,s  = np.linalg.lstsq(factor2.T,obs1.T)
        #Fitting = np.dot(factor2.T,Coef)
        Regression.fit(factor2.T,obs_Std)
        Fitting = Regression.predict(factor2.T)
        obs_hindcast = mapstd(Fitting,obs_PS_Std,stat='reverse')
        Fitting = np.ravel(obs_hindcast)

        PS,SC,ACC,RMSE=do_PS(Fitting,obs1,20,50)
        #factor3 = dclimate.mapstd(factor0,PS0,stat='apply')
        ## print(u'|因子位置=',sel1,factor_p.shape,end='')
        #sys.exit(0)
        #factor3 = factor_p[sel1,:]
        self.selecter = sel1
        self.coef = Regression.coef_
        self.hindcast = obs_hindcast
        self.Regression = Regression
        self.obs_PS_Std = obs_PS_Std
        self.factor_PS_h = PS_h
        return

    def predict(self,factor_p):
        factor_p = mapstd(factor_p,self.factor_PS_h,stat='apply')
        factor3 = factor_p[self.selecter,:]
        #Pred = np.dot(factor3.T,Coef)
        Pred = self.Regression.predict(factor3.T)
        print(u'|因子系数=',self.Regression.coef_,end='')
        print(u'|所选因子序列为',self.selecter)
        #sys.exit(0)

        Pred = mapstd(Pred,self.obs_PS_Std,stat='reverse')
        Pred = np.ravel(Pred)
        print(Pred)
        #sys.exit(0)
        ## print(u'预报值 = ',Pred,Pred.shape)
        return Pred


