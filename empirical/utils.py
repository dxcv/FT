# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  11:00
# NAME:FT_hp-utils.py
import numpy as np
import scipy
from scipy import stats
import pandas as pd
from sklearn.decomposition import PCA


def GRS_test(factor,resid,alpha):
    '''
        returns the GRS test statistic and its corresponding p-value proposed in
    Gibbons,Ross,Shanken (1989),to test the hypothesis:alpha1=alpha2=...=alphaN=0.
    That is if the alphas from time series Regression on N Test Assets are
    cummulatively zero.

    Args:
        factor:TxL Matrix of factor returns (the "Matrix" means numpy.matrixlib.defmatrix.matrix)
        resid: TxN Matrix of residuals from TS-regression
        alpha: Nx1 Matrix of intercepts from TS-regression

    details:
        N:the number of assets (or the number of regressions)
        T:the number of the observations
        L:the number of the factors

    Returns:
        GRS[0,0]:1x1 scalar of GRS Test-value
        pvalue[0,0]:1x1 scalar of P-value from an F-Distribution

    '''
    #check the type of the input data
    for _d in [factor,resid,alpha]:
        if not isinstance(_d,np.matrix):
            raise TypeError('{} is not a np.matrix'.format(_d))

    T, N = resid.shape
    L = factor.shape[1]
    if T-N-L<=0:
        raise ValueError('T-N-L<0, can not conduct F test')

    mu_mean = np.mat(factor.mean(0)).reshape(L,1)  # Lx1 mean excess factor returns
    cov_e=np.cov(resid.T).reshape(N,N)
    cov_f=np.cov(factor.T).reshape(L,L)
    GRS = (T * 1.0 / N) * ((T - N - L) * 1.0 / (T - L - 1)) \
          * (alpha.T * np.linalg.inv(cov_e) * alpha)\
          / (1 + mu_mean.T * np.linalg.inv(cov_f) * mu_mean)
    cdf = stats.f.cdf(GRS, N, (T - N - L))
    return GRS[0,0],1-cdf[0,0]

def run_GRS_test(model,asset):
    '''

    Args:
        model:DataFrame,(T,L) do not contain intercept
        asset:DataFrame,(T,N) asset to pricing,for example, it can be a set of portfolios sorted by size

    Returns:

    '''
    comb=pd.concat([model,asset],axis=1,join='inner')
    comb=comb.dropna()

    a=np.array(comb.loc[:,model.columns])
    A=np.hstack([np.ones([len(a),1]),a])

    resid=[]
    alpha=[]
    for col in asset.columns:
        y=comb.loc[:,col]
        beta=np.linalg.lstsq(A,y,rcond=None)[0]
        res=y-np.dot(A,beta)
        al=beta[0]

        resid.append(res)
        alpha.append(al)
    resid=pd.concat(resid,axis=1)

    factor=np.matrix(a)
    resid=np.matrix(resid.values)
    alpha=np.matrix(alpha).T

    grs,p=GRS_test(factor,resid,alpha)
    return grs,p


def unify_index(df1,df2):
    common=df1.index.intersection(df2.index)
    return df1.reindex(index=common),df2.reindex(index=common)

def align_index(df1,df2):
    ind=df1.index.intersection(df2.index)
    return df1.loc[ind],df2.loc[ind]


def get_pca_factors(df,n):
    '''

    Args:
        df:DataFrame
        n:

    Returns:

    '''
    X=df.values
    pca=PCA(n_components=n)
    pca_factors=pd.DataFrame(pca.fit_transform(X),index=df.index,
                             columns=['pca{}'.format(i) for i in range(1,n+1)])
    return pca_factors

