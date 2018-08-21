# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-17  00:29
# NAME:FT_hp-HaiTong.py
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from data.dataApi import read_local


DIR=r'G:\FT_Users\HTZhang\haitong'



def outlier(s, k=4.5):
    '''
    Parameters
    ==========
    s:Series
        原始因子值
    k = 3 * (1 / stats.norm.isf(0.75))
    '''
    med = np.median(s) #debug: NaN should be removed before apply this function
    mad = np.median(np.abs(s - med))
    uplimit = med + k * mad
    lwlimit = med - k * mad
    y = np.where(s >= uplimit, uplimit, np.where(s <= lwlimit, lwlimit, s))
    # return pd.DataFrame(y, index=s.index)
    return pd.Series(y, index=s.index)

def z_score(x):
    return (x - np.mean(x)) / np.std(x)


indNames=['size','log_size','bp','cumRet','turnover','amihud','ivol']

def clean_indicators():
    for indname in indNames:
        s=pd.read_pickle(os.path.join(DIR,indname+'.pkl')).stack().swaplevel()
        s.name=indname

        fdmt_m = read_local('fdmt_m')
        data = pd.concat([fdmt_m, s], axis=1, join='inner')
        data.index.names=['stkcd','month_end']

        data = data.dropna(subset=['type_st', 'young_1year'])
        data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
        data=data.dropna(subset=[indname])
        data[indname+'_out']=data.groupby('month_end')[indname].apply(outlier)
        data[indname+'_zsc']=data.groupby('month_end')[indname+'_out'].apply(z_score)
        cleaned=pd.pivot_table(data,indname+'_zsc','month_end','stkcd')
        cleaned.to_pickle(os.path.join(DIR,'standardized',indname+'.pkl'))
        print(indname)


def get_factor_loading():
    dfs = []
    for indname in indNames:
        _df = pd.read_pickle(
            os.path.join(DIR, 'standardized', indname + '.pkl'))
        if indname is not 'cumRet':
            _df = _df.ffill()
        _df = _df.dropna(axis=0, how='all').stack()
        dfs.append(_df)

    factor_loading = pd.concat(dfs, axis=1, keys=indNames, join='inner')
    factor_loading.index.names = ['month_end', 'stkcd']
    factor_loading.to_pickle(os.path.join(DIR,'factor_loading.pkl'))

def reg(df):
    y=df['ret']
    X=df[['const']+indNames]
    r=sm.OLS(y,X).fit()
    return r

def get_factor_return():
    factor_loading=pd.read_pickle(os.path.join(DIR,'factor_loading.pkl'))
    comb = factor_loading.groupby('stkcd').shift(1)  # trick:use the indicator of time t-1 to regress on return of time t
    ret = pd.read_pickle(os.path.join(DIR, 'cumRet.pkl')).stack()
    comb['ret'] = ret
    comb = comb.dropna()
    comb.to_pickle(os.path.join(DIR, 'comb.pkl'))
    comb=sm.add_constant(comb)
    rs=comb.groupby('month_end').apply(reg)
    factor_return=pd.concat([r.params[1:] for r in rs],axis=1,keys=rs.index).T
    factor_return.to_pickle(os.path.join(DIR,'factor_return.pkl'))


factor_loading=pd.read_pickle(os.path.join(DIR,'factor_loading.pkl'))
factor_return=pd.read_pickle(os.path.join(DIR,'factor_return.pkl'))

# factor_return.cumsum().plot().get_figure().show()

WINDOW=12
factor_return_predicted=factor_return.rolling(WINDOW).mean() #TODO: 12

groups=factor_loading.groupby('month_end').groups

t=list(groups.keys())[20]
df=factor_loading.groupby('month_end').get_group(t)# factor loading at time t

fr=factor_return_predicted.loc[t] #predicted factor return for t+1

mu=np.matrix(df) @ np.matrix(fr).T


#zz500_ret_d = read_local('equity_selected_indice_ir')['zz500_ret_d']

from cvxopt import solvers

P=0
q=-mu
G=np.vstack((np.eye(len(mu)),-np.eye(len(mu))))
h=np.vstack((np.matrix([[0.01]]*len(mu)),np.matrix([[0]]*len(mu))))

A=np.matrix([1]*len(mu))
b=1

sol=solvers.qp(P,q,G,h,A,b)

