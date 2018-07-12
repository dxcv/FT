# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  10:57
# NAME:FT-tools.py

import time
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np


def monitor(func):
    def wrapper(*args,**kwargs):
        print('{}   starting -> {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),
                                           func.__name__))
        return func(*args,**kwargs)
    return wrapper

def get_inter_index(s1, s2):
    interInd=s1.index.intersection(s2.index)
    s1=s1.reindex(interInd)
    s2=s2.reindex(interInd)
    return s1, s2

def _convert_freq(x,freq,thresh):
    x=x.groupby(pd.Grouper(freq=freq,level='trd_dt')).last()
    # TODO: ffill whould only be used on indicators from financial report.
    # TODO: pay attention to cash_div and counts in
    x=x.ffill(limit=thresh)
    return x

def convert_freq(x, freq='M', thresh=12):
    newdf=x.groupby('stkcd').apply(_convert_freq, freq, thresh)
    newdf=newdf.swaplevel().sort_index()
    return newdf

def handle_duplicates(df):
    return df[~df.index.duplicated(keep='first')]

def number2dateStr(x):
    if x:
        if isinstance(x,(int,float)):
            x=str(x)

        if '.' in x:
            x=x.split('.')[0]
        return x

def daily2monthly(df):
    '''

    Args:
        df:DataFrame,contains column ['stkcd','trd_dt']

    Returns:DataFrame, only add a new column named 'month_end' to the input df

    '''
    df=df.sort_values(['stkcd','trd_dt'])
    monthly=df[(df['stkcd']==df['stkcd'].shift(-1)) &
                     (df['trd_dt'].dt.month!=df['trd_dt'].shift(-1).dt.month)]
    monthly=monthly.dropna(how='all')
    monthly['month_end']=monthly['trd_dt']+MonthEnd(0)
    return monthly

def filter_st_and_young(df,fdmt_m):
    data=pd.concat([fdmt_m,df],axis=1).reindex(fdmt_m.index)
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    return data

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

def neutralize(df, col, industry, cap='ln_cap'):
    '''
    Parameters
    ===========
    df:
        包含标准化后的因子值的DataFrame
    industry: list of industry columns
        排除第一行业代码后的m-1个行业代码

    Returns
    =======
    res:
        标准化因子对行业哑变量矩阵和对数市值回归后的残差
    '''
    a = np.array(df.loc[:, industry + [cap]])
    A = np.hstack([a, np.ones([len(a), 1])])
    y = df.loc[:, col]
    beta = np.linalg.lstsq(A, y,rcond=None)[0]
    res = y - np.dot(A, beta)
    return res

def clean(df, col,by='month_end'):
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
    df[col + '_out']=df.groupby(by)[col].apply(outlier)
    df[col + '_zsc']=df.groupby(by)[col + '_out'].apply(z_score)
    df['wind_2'] = df['wind_indcd'].apply(str).str.slice(0, 6) # wind 2 级行业代码
    df = df.join(pd.get_dummies(df['wind_2'], drop_first=True))
    df['ln_cap'] = np.log(df['cap'])
    industry = list(np.sort(df['wind_2'].unique()))[1:]
    df[col + '_neu'] = df.groupby(by, group_keys=False).apply(neutralize, col + '_zsc', industry)

    del df[col]
    del df[col + '_out']
    del df[col + '_zsc']
    df=df.rename(columns={col + '_neu':col})
    return df
