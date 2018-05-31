# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-27  22:07
# NAME:FT-base_function.py
from config import DCSV, DPKL
import os
import pandas as pd
import numpy as np

# refer to the paper for operator
from singleFactor.factors.check import check_factor
from tools import handle_duplicates
'''
Notes:
1. before cal pct_change,we should delete the items with negative indicators,
or we will get opposite value.



'''
#TODO: 计算的时候用['stkcd','report_period']作为index，在test 的时候用['stkcd',
#TODO: 'trd_dt'] 作为index

# TODO：到最后test的时候才调用 _change_index()




def ttm_adjust(s):
    return s.groupby('stkcd').apply(
        lambda x:x.rolling(4,min_periods=4).sum())

#------------------------------------base function-----------------------------
'''
input must be DataFrame contained 'trd_dt',since we will set 'trd_dt' as index in 
the function '_change_index'.

the output is a series without name,and the index is ['stkcd','report_period']
'''

def raw_level(df, col,name, ttm=True):
    '''
    计算某个level的ttm
    df 是按季度的数据
    '''
    df[name]=df[col]
    if ttm:
        df[name]=ttm_adjust(df[name])
    return df

#TODO: test whether the raw_level function will change the data

def x_pct_chg(df, col, name,q=1, ttm=True,delete_negative=True):
    '''
        d(x)/x
    percentage change in each accounting variable

    Args:
        df:按季度的数据
        col:
        q:int,季度数，如果是求同比增长，则取q=4
        ttm: True or False

    Returns: df[['result']]

    '''
    df[name]=df[col]
    if delete_negative:
        df[name][df[name]<=0]=np.nan
    if ttm:
        df[name]=ttm_adjust(df[name])
    df[name]=df[name].groupby('stkcd').apply(lambda s:s.pct_change(periods=q))
    return df

def x_history_growth_avg(df,col,name,q=12,ttm=False,delete_negative=True):
    df[name]=df[col]
    if delete_negative:
        df[name][df[name]<=0]=np.nan
    if ttm:
        df[name]=ttm_adjust(df[name])

    def _cal(s,q):
        pct_chg=s.pct_change()
        return pct_chg.rolling(q,min_periods=int(q/2)).mean()

    df[name]=df[name].groupby('stkcd').apply(_cal,q)
    return df

def x_history_compound_growth(df, col,name, q=20, ttm=True, delete_negative=True):
    '''
    计算过去q个季度的复合增长率
    Args:
        df:
        col:
        q:
        ttm:
        delete_negative:

    Returns:

    '''
    df[name]=df[col]
    if delete_negative:
        df[name][df[name]<=0]=np.nan
    if ttm:
        df[name]=ttm_adjust(df[name])

    def _cal_cumulative_g(arr):
        return np.cumprod((np.diff(arr)/arr[:-1])+1)[-1]-1

    df[name]=df[name].groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).apply(_cal_cumulative_g))
    return df

def x_history_std(df,col,name,q=8,ttm=True):
    '''
    std(x,q)
    Args:
        df:
        col:
        q: int,quarters
        ttm: boolean

    Returns: pd.Series

    '''
    df[name]=df[col]
    if ttm:
        df[name]=ttm_adjust(df[name])
    df[name]=df[name].groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).std())
    return df

def x_history_downside_std(df,col,name,q=8,ttm=False):
    '''
    stddev(min(x-x(-1),0))

    #TODO:normalize (scale with the mean value)
    Args:
        df:
        col:
        q:
        ttm:

    Returns:

    '''
    def downside_risk(s, q):
        dev = s - s.shift(1)
        downside = dev.where(dev < 0, 0)
        r = downside.rolling(q, min_periods=q).std()
        return r

    df[name]=df[col]
    if ttm:
        df[name]=ttm_adjust(df[name])
    df[name]=df[name].groupby('stkcd').apply(downside_risk,q)
    return df

def ratio_x_y(df, col1, col2,name,ttm=True,delete_negative_y=True):
    '''
    x/y
    financial ratio in x/y
    '''
    data=df.copy()
    if delete_negative_y:
        data[col2][data[col2]<=0]=np.nan
    if ttm:
        data[col1]=ttm_adjust(data[col1])
        data[col2]=ttm_adjust(data[col2])
    data['x']=data[col1]
    data['y']=data[col2]

    data[name]=data['x']/data['y']
    return data

def ratio_yoy_chg(df, col1, col2, name,ttm=True,delete_negative_y=True):
    '''
    d(x/y)
    year-to-year change in financial ratio
    '''
    data=df.copy()
    data['x']=data[col1]
    data['y']=data[col2]
    if delete_negative_y:
        data['y'][data['y']<=0]=np.nan
    if ttm:
        data['x']=ttm_adjust(data['x'])
        data['y']=ttm_adjust(data['y'])
    data['ratio']=data['x']/data['y']
    data[name]=data['ratio'].groupby('stkcd').apply(
        lambda s:s-s.shift(4))
    return data

def ratio_yoy_pct_chg(df, col1, col2,name, ttm=True,delete_negative_y=True):
    '''
    d(x/y)/(x/y)
    year-to-year "percent" change in financial ratio
    '''
    data=df.copy()
    data['x']=data[col1]
    data['y']=data[col2]

    if delete_negative_y:
        data['y'][data['y']<=0]=np.nan
    if ttm:
        data['x']=ttm_adjust(data['x'])
        data['y']=ttm_adjust(data['y'])
    data['ratio']= data['x']/data['y']

    data[name]=data['ratio'].groupby('stkcd').apply(
        lambda s:s.pct_change(periods=4))
    return data

def pct_chg_dif(df, col1, col2,name, ttm=True,delete_negative=True):
    '''
    d(x)/x -d(y)/y
    the difference between the percentage change in each accounting variable and
    the percentage change in a base variable
    '''
    data=df.copy()
    data['x']=data[col1]
    data['y']=data[col2]
    if delete_negative:
        data['x'][data['x']<=0]=np.nan
        data['y'][data['y']<=0]=np.nan
    if ttm:
        data['x']=ttm_adjust(data['x'])
        data['y']=ttm_adjust(data['y'])

    data['pct_chg_x']=data.x.groupby('stkcd').apply(
        lambda s:s.pct_change())
    data['pct_chg_y']=data['y'].groupby('stkcd').apply(
        lambda s:s.pct_change())
    data[name]=data['pct_chg_x']-data['pct_chg_y']
    return data


def ratio_x_chg_over_lag_y(df, col1, col2,name, ttm=True,delete_negative_y=True):
    '''
    d(x)/lag(y)
    the change in each accounting variable scaled by a lagged base variable
    '''
    data=df.copy()
    data['x']=data[col1]
    data['y']=data[col2]
    if delete_negative_y:
        data['y'][data['y']<=0]=np.nan

    if ttm:
        data['x'] = ttm_adjust(data['x'])
        data['y'] = ttm_adjust(data['y'])
    data['x_chg']=data['x'].groupby('stkcd').apply(lambda s: s - s.shift(1))
    data['lag_y']=data['y'].groupby('stkcd').shift(1)
    data[name]=data['x_chg']/data['lag_y']
    return data
