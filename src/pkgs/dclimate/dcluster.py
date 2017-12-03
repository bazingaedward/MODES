# -*- coding: cp936 -*-
import sys
from scipy.spatial import distance
from sklearn import cluster, datasets
from sklearn.metrics import euclidean_distances
from scipy.stats import pearsonr
import numpy as np
import dclimate as dclim

from geopy.distance import great_circle

def distance_great_circle(X,Y):
    X=(X)
    Y=(Y)
    #print('X,Y=',X,Y)
    #print(great_circle(X, Y).km)
    #sys.exit(0)
    return great_circle(X, Y).km

def distance_dtw(X,Y):
    from dtw import dtw
    X=(X)
    Y=(Y)
    #print('X,Y=',X,Y)
    #print(great_circle(X, Y).km)
    #sys.exit(0)
    distace1,_,_ = dtw(X,Y)
    return distace1


def distance_sign(X,Y):
    X=X.flatten()
    Y=Y.flatten()
    Z=X*Y
    Z=np.where(Z>0,1.0,0.0)
    return np.mean(Z)

def distance_pearson_corr(X,Y):
    X=X.flatten()
    Y=Y.flatten()
    a,b = pearsonr(X,Y)
    return a


def distance_pearson_corr_r(X,Y):
    X=X.flatten()
    Y=Y.flatten()
    a,b = pearsonr(X,Y)
    return a

def distance_pearson_corr_p(X,Y):
    X=X.flatten()
    Y=Y.flatten()
    a,b = pearsonr(X,Y)
    return b


###########################
def apcluster_k(S,K,lam=0.5):
    p_Max=0
    p_Min=0
    p=np.median(S)
    idx,k=apclusterd(S,p,lam)
    p_old = p #此处为负值

    if(0==K):
        idx,k=apclusterd(S,p,lam)
        return idx,k


    while(k!=K):
        idx,k=apclusterd(S,p,0.7)
        print('P=%f,k=%d,p_Max=%f,p_Min=%f'%(p,k,p_Max,p_Min))
        if(k<K):
            print('k<%d'%K)
            p=p*0.8  #逐步增大但是不大于0
            p_Max=p
        else:
            #if(p_Max<0 and p_Min<0):
            #    P=p_Max*1.1
            #    continue
            print('k>=%d'%K)
            p=p+p*0.1
            p_Min=p
    return idx,k
#################################

def apclusterd(S,P=0,lam=0.5):
    #迭代开始

    if(0==P):
        P=np.median(S)

    N = S.shape[0]
    for ii in range(N):
        S[ii,ii]=P
        #------------------------
    R = np.zeros((N,N))  #R为适合度
    A = np.zeros((N,N))  #A为响应度
    #------------------------
    for iter  in range(100):
        #print(iter)
        #Compute responsibilities
        R_old=R
        AS=A+S
        #重要费了一番功夫
        I=AS.argmax(1);Y=AS[np.arange(0,N,1),I]
        for i in range(N):
            AS[i,I[i]]=-np.finfo(np.float).max
            #----------------
        Y2=AS.max(1)
        R = S - np.tile(Y, (N, 1)).T
        #----------------
        for i in range(N):
            R[i,I[i]]=S[i,I[i]]-Y2[i]
        R=(1-lam)*R+lam*R_old #Dampen responsibilities 阻尼响应

        # Compute availabilities
        A_old=A
        Rp=np.maximum(R,0)

        for k in range(N):
            Rp[k,k]=R[k,k]

        A=np.tile(sum(Rp,0), (N, 1))-Rp
        dA = np.diag(A)
        A=np.minimum(A,0)

        for k in range(N):
            A[k,k]=dA[k]

        A=(1-lam)*A+lam*A_old #Dampen availabilities 阻尼有效值
        #print(A)

    E=R+A; # Pseudomarginals

    #print(E)

    I=np.nonzero(np.diag(E)>0)
    #print(I)
    I=I[0]

    K=len(I)
    #S2 = S[:,I]
    #print('SI=',S[:,I])
    c=S[:,I].argmax(1)
    #print('c=',c)
    c[I]=np.arange(0,K,1)
    #print('c=',c)
    #tmp = S2[np.arange(0,N,1),c]
    idx = I[c]  #聚类的点
    #print('index =',idx)
    #print('cluster number=',K)
    return idx,K
    '''
    '''
    #print(tmp)

#tmp=AS[np.arange(0,N,1),I]
#sys.exit(0)
#K=length(I) # Indices of exemplars
#[tmp c]=max(S(:,I),[],2); c(I)=1:K;
#idx=I(c) % Assignments

if __name__ == "__main__" :
    N=2500
    x=np.random.rand(N,2)
    #print(x)
    #sys.exit(0)
    if(0):
        x=np.array([0,0, \
                    .5, 0, \
                    1, 0, \
                    0, 2, \
                    .25, 2, \
                    0.5 ,2])
        x = x.reshape(-1,2)

    #sys.exit(0)
    #x=rand(N,2); M=N*N-N; s=zeros(M,3); j=1;

    s =  - distance.pdist(x,'euclidean');
    S = distance.squareform(s)
    #print(S)
    P= np.median(s)


    #print(S)
    #print('*'*80)
    #S=S+1e-12*np.random.randn(N,N)*(np.max(S.flatten())-np.min(S.flatten()))
    #print(S)
    #sys.exit(0)
    # Remove degeneracies  删除退化？
    lam=0.5; # Set damping factor
    print(x)
    print('N=',N)
    mytic()
    apclusterd(S,P,lam)
    mytoc('aa')
