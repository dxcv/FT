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


the output is a DataFrame with index as ['stkcd','report_period'],and the factor
is stored in the column named 'target'
'''

def raw_level(df, col,ttm=True):
    '''
    计算某个level的ttm
    df 是按季度的数据
    '''
    df=df.copy()
    if ttm:
        df[col]=ttm_adjust(df[col])
    df['target']=df[col]
    return df

#TODO: test whether the raw_level function will change the data

def x_pct_chg(df, col, q=1, ttm=True,delete_negative=True):
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
    df=df.copy()
    if delete_negative:
        df[col]=df[col].where(df[col]>0,np.nan)
        # df[col][df[col]<=0]=np.nan
    if ttm:
        df[col]=ttm_adjust(df[col])
    df['target']=df[col].groupby('stkcd').apply(lambda s:s.pct_change(periods=q))
    return df

def x_history_growth_avg(df, col, q=12, ttm=False, delete_negative=True):
    df=df.copy()
    if delete_negative:
        df[col]=df[col].where(df[col]>0,np.nan)
        # df[col][df[col] <= 0]=np.nan
    if ttm:
        df[col]=ttm_adjust(df[col])

    def _cal(s,q):
        pct_chg=s.pct_change()
        return pct_chg.rolling(q,min_periods=int(q/2)).mean()

    df['target']=df[col].groupby('stkcd').apply(_cal, q)
    return df

def x_history_compound_growth(df, col, q=20, ttm=True, delete_negative=True):
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
    df=df.copy()
    if delete_negative:
        df[col]=df[col].where(df[col]>0,np.nan)
        # df[col][df[col] <= 0]=np.nan
    if ttm:
        df[col]=ttm_adjust(df[col])

    def _cal_cumulative_g(arr):
        return np.cumprod((np.diff(arr)/arr[:-1])+1)[-1]-1

    df['target']=df[col].groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).apply(_cal_cumulative_g))
    return df

def x_history_std(df, col, q=8, ttm=True):
    '''
    std(x,q)
    Args:
        df:
        col:
        q: int,quarters
        ttm: boolean

    Returns: pd.Series

    '''
    df=df.copy()
    if ttm:
        df[col]=ttm_adjust(df[col])
    df['target']=df[col].groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).std())
    return df

def x_history_downside_std(df, col, q=8, ttm=False):
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

    df=df.copy()
    if ttm:
        df[col]=ttm_adjust(df[col])
    df['target']=df[col].groupby('stkcd').apply(downside_risk, q)
    return df

def ratio_x_y(df, col1, col2,ttm=True,delete_negative_y=True):
    '''
    x/y
    financial ratio in x/y
    '''
    df=df.copy()
    if delete_negative_y:
        df[col2]=df[col2].where(df[col2]>0,np.nan)
        # df[col2][df[col2]<=0]=np.nan
    if ttm:
        df[col1]=ttm_adjust(df[col1])
        df[col2]=ttm_adjust(df[col2])
    df['x']=df[col1]
    df['y']=df[col2]

    df['target']=df['x']/df['y']
    return df

def ratio_yoy_chg(df, col1, col2,ttm=True,delete_negative_y=True):
    '''
    d(x/y)
    year-to-year change in financial ratio
    '''
    df=df.copy()
    df['x']=df[col1]
    df['y']=df[col2]
    if delete_negative_y:
        df['y']=df['y'].where(df['y']>0,np.nan)
        # df['y'][df['y']<=0]=np.nan
    if ttm:
        df['x']=ttm_adjust(df['x'])
        df['y']=ttm_adjust(df['y'])
    df['ratio']=df['x']/df['y']
    df['target']=df['ratio'].groupby('stkcd').apply(
        lambda s:s-s.shift(4))
    return df

def ratio_yoy_pct_chg(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x/y)/(x/y)
    year-to-year "percent" change in financial ratio
    '''
    df=df.copy()
    df['x']=df[col1]
    df['y']=df[col2]

    if delete_negative_y:
        df['y']=df['y'].where(df['y']>0,np.nan)
        # df['y'][df['y']<=0]=np.nan
    if ttm:
        df['x']=ttm_adjust(df['x'])
        df['y']=ttm_adjust(df['y'])
    df['ratio']= df['x']/df['y']

    df['target']=df['ratio'].groupby('stkcd').apply(
        lambda s:s.pct_change(periods=4))
    return df

def pct_chg_dif(df, col1, col2, ttm=True,delete_negative=True):
    '''
    d(x)/x -d(y)/y
    the difference between the percentage change in each accounting variable and
    the percentage change in a base variable
    '''
    df=df.copy()
    df['x']=df[col1]
    df['y']=df[col2]
    if delete_negative:
        df['x']=df['x'].where(df['x']>0,np.nan)
        df['y']=df['y'].where(df['y']>0,np.nan)
    if ttm:
        df['x']=ttm_adjust(df['x'])
        df['y']=ttm_adjust(df['y'])

    df['pct_chg_x']=df.x.groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['pct_chg_y']=df['y'].groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['target']=df['pct_chg_x']-df['pct_chg_y']
    return df

def ratio_x_chg_over_lag_y(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x)/lag(y)
    the change in each accounting variable scaled by a lagged base variable
    '''
    df=df.copy()
    df['x']=df[col1]
    df['y']=df[col2]
    if delete_negative_y:
        df['y']=df['y'].where(df['y']>0,np.nan)
    if ttm:
        df['x'] = ttm_adjust(df['x'])
        df['y'] = ttm_adjust(df['y'])
    df['x_chg']=df['x'].groupby('stkcd').apply(lambda s: s - s.shift(1))
    df['lag_y']=df['y'].groupby('stkcd').shift(1)
    df['target']=df['x_chg']/df['lag_y']
    return df
