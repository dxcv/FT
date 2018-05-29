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



base_variables=[]

def ttm_adjust(s):
    return s.groupby('stkcd').apply(
        lambda x:x.rolling(4,min_periods=4).sum())

def adjust_result_format(df):
    '''
    虽然用了stkcd和report_period 作为主键，但是不同的report_period 对应的trd_dt
    可能相同，比如，asharefinancialindicator 中的000002.SZ，其2006-12-31 和
    2007-12-31的trd_dt 都是 2008-03-21
    '''
    df=df.reset_index().sort_values(['stkcd','trd_dt','report_period'])
    df=df[~df.duplicated(['stkcd','trd_dt'],keep='last')]
    df=df.set_index(['trd_dt','stkcd']).sort_index()[['result']].dropna()
    return df

def raw_level(df, col, ttm=True):
    '''
    计算某个level的ttm
    df 是按季度的数据
    '''
    x=df[col]
    if ttm:
        x=ttm_adjust(x)
    df['result']=x
    return adjust_result_format(df)

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
    x=df[col]
    if delete_negative:
        x[x<0]=np.nan
    if ttm:
        x=ttm_adjust(x)
    df['result']=x.groupby('stkcd').apply(lambda s:s.pct_change(periods=q))
    return adjust_result_format(df)

def x_compound_growth(df, col, q=60, ttm=True, delete_negative=True):
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
    x=df[col]
    if delete_negative:
        x[x<0]=np.nan
    if ttm:
        x=ttm_adjust(x)

    def _cal_cumulative_g(arr):
        return np.cumprod((np.diff(arr)/arr[:-1])+1)[-1]-1

    df['result']=x.groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).apply(_cal_cumulative_g))
    return adjust_result_format(df)

def ratio_x_y(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    x/y
    financial ratio in x/y
    '''
    x = df[col1]
    y = df[col2]

    if delete_negative_y:
        y[y<0]=np.nan

    if ttm:
        x=ttm_adjust(x)
        y=ttm_adjust(y)

    df['result']=x/y
    return adjust_result_format(df)

def ratio_yoy_chg(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x/y)
    year-to-year change in financial ratio
    '''
    x=df[col1]
    y=df[col2]

    if delete_negative_y:
        y[y<0]=np.nan

    if ttm:
        x=ttm_adjust(x)
        y=ttm_adjust(y)
    df['ratio']=x/y
    df['result']=df['ratio'].groupby('stkcd').apply(
        lambda s:s-s.shift(4))
    return adjust_result_format(df)

def ratio_yoy_pct_chg(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x/y)/(x/y)
    year-to-year "percent" change in financial ratio
    '''
    x=df[col1]
    y=df[col2]

    if delete_negative_y:
        y[y<0]=np.nan

    if ttm:
        x=ttm_adjust(x)
        y=ttm_adjust(y)
    df['ratio']= x/y
    df['result']=df['ratio'].groupby('stkcd').apply(
        lambda s:s.pct_change(periods=4))
    return adjust_result_format(df)

def pct_chg_dif(df, col1, col2, ttm=True,delete_negative=True):
    '''
    d(x)/x -d(y)/y
    the difference between the percentage change in each accounting variable and
    the percentage change in a base variable
    '''
    x=df[col1]
    y=df[col2]

    if delete_negative:
        x[x<0]=np.nan
        y[y<0]=np.nan
    if ttm:
        x=ttm_adjust(x)
        y=ttm_adjust(y)

    df['pct_chg_x']=x.groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['pct_chg_y']=y.groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['result']=df['pct_chg_x']-df['pct_chg_y']
    return adjust_result_format(df)

def ratio_x_chg_over_lag_y(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x)/lag(y)
    the change in each accounting variable scaled by a lagged base variable
    '''
    x=df[col1]
    y=df[col2]
    if delete_negative_y:
        y[y<0]=np.nan

    if ttm:
        x=ttm_adjust(x)
        y=ttm_adjust(y)
    df['x_chg']=x.groupby('stkcd').apply(lambda s: s - s.shift(1))
    df['lag_y']=y.groupby('stkcd').shift(1)
    df['result']=df['x_chg']/df['lag_y']
    return adjust_result_format(df)




