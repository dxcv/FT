# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-29  16:09
# NAME:FT-dive_into_mom1.py


'''
y  ret_1m


x:
ln_cap
bp
roe
oper_rev growth q=4
mom 1m
mom 12
industry

sqrt(cap) weighted

'''
import os

import pandas as pd
from data.dataApi import read_local
import numpy as np
from pandas.tseries.offsets import MonthEnd
from pyparsing import Word


def outlier(x, k=4.5):
    '''
    Parameters
    ==========
    x:
        原始因子值
    k = 3 * (1 / stats.norm.isf(0.75))
    '''
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    uplimit = med + k * mad
    lwlimit = med - k * mad
    y = np.where(x >= uplimit, uplimit, np.where(x <= lwlimit, lwlimit, x))
    return pd.DataFrame(y, index=x.index)

def z_score(x):
    return (x - np.mean(x)) / np.std(x)

def clean(df, col):
    '''
    Parameters
    ==========
    df: DataFrame
        含有因子原始值、市值、行业代码
    col:
        因子名称
    '''

    # Review: 风格中性：对市值对数和市场做回归后取残差
    #TODO： 市值中性化方式有待优化，可以使用SMB代替ln_cap
    df[col + '_out']=df.groupby('month_end')[col].apply(outlier)
    df[col + '_zsc']=df.groupby('month_end')[col + '_out'].apply(z_score)
    return df[col+'_zsc']

def daily2monthly(daily):
    daily=daily.reset_index()
    daily=daily.sort_values(['stkcd','trd_dt'])

    daily.where((daily['stkcd']==daily['stkcd'].shift(-1)) & (daily['trd_dt'].dt.month!= daily['trd_dt'].shift(-1).dt.month) )

    monthly=daily.groupby('stkcd').resample('M',on='trd_dt').last().dropna()
    del monthly['trd_dt']
    monthly.index.names=['stkcd','month_end']
    return monthly

indicators=['T__mom_1M','T__mom_12M','G_pct_4__tot_oper_rev','V__bp','Q__roe']
directory=r'D:\zht\database\quantDb\internship\FT\singleFactor\indicators'
fdmt = read_local('fdmt_m')[
    ['cap', 'type_st', 'wind_indcd', 'young_1year']]

ret_1m = read_local('trading_m')['ret_1m']
dfs=[]
for indicator in indicators:
    df=pd.read_pickle(os.path.join(directory,indicator+'.pkl'))
    dfs.append(df)

data=pd.concat([fdmt,ret_1m]+dfs,axis=1).reindex(fdmt.index)
data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
data=data.groupby('stkcd').ffill().dropna()

for indicator in indicators:
    print(indicator)
    data[indicator]=clean(data,indicator)

data=data.groupby('month_end').filter(lambda x:x.shape[0]>300)
data['sqrt_cap']=np.sqrt(data['cap'])
data['wind_2'] = data['wind_indcd'].apply(str).str.slice(0, 6)

def reg(data,cap_weight=True):
    industry = list(np.sort(data['wind_2'].unique()))[1:]
    data = data.join(pd.get_dummies(data['wind_2'], drop_first=True))
    a=data[['T__mom_1M', 'T__mom_12M', 'G_pct_4__tot_oper_rev','V__bp', 'Q__roe', 'sqrt_cap']+industry].values
    A = np.hstack([a, np.ones([len(a), 1])])
    y=data[['ret_1m']].values

    if cap_weight:
        W=np.sqrt(np.diag(data['cap'].values))
        AW=np.dot(W,A)
        yW=np.dot(W,y)
        beta=np.linalg.lstsq(AW,yW)[0][0][0]
    else:
        beta = np.linalg.lstsq(A, y, rcond=None)[0][0][0]
    return beta

betas=data.groupby('month_end').apply(reg,False)
betas_weighted=data.groupby('month_end').apply(reg,True)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 8))
ax1 = plt.subplot(211)
ax1.bar(betas.index, betas.values, width=20, color='b')
ax1.set_ylabel('return bar')
ax1.set_title('equal weighted')

ax4 = ax1.twinx()
ax4.plot(betas.index, betas.cumsum(), 'r-')
ax4.set_ylabel('cumsum', color='r')
[tl.set_color('r') for tl in ax4.get_yticklabels()]

ax2 = plt.subplot(212)
ax2.bar(betas_weighted.index, betas_weighted.values, width=20, color='b')
ax2.set_ylabel('return bar')
ax2.set_title('cap weighted')

ax3 = ax2.twinx()
ax3.plot(betas_weighted.index, betas_weighted.cumsum(), 'r-')
ax3.set_ylabel('cumsum', color='r')
[tl.set_color('r') for tl in ax3.get_yticklabels()]

fig.savefig(r'e:\a\fig.png')


