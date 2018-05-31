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

# TODO：到最后test的时候才调用 _adjust_result_format()




def ttm_adjust(s):
    return s.groupby('stkcd').apply(
        lambda x:x.rolling(4,min_periods=4).sum())

def _adjust_result_format(df):
    '''
        虽然用了stkcd和report_period 作为主键，但是不同的report_period 对应的trd_dt
    可能相同，比如，asharefinancialindicator 中的000002.SZ，其2006-12-31 和
    2007-12-31的trd_dt 都是 2008-03-21
    Args:
        df:

    Returns:DataFrame,the index is ['stkcd','trd_dt']

    '''
    if isinstance(df['trd_dt'],pd.DataFrame):
        '''如果df里边有多个名为trd_dt的列，取日期最大的那个'''
        trd_dt_df=df['trd_dt']
        df=df.drop('trd_dt',axis=1)
        df['trd_dt']=trd_dt_df.max(axis=1)

    df=df.reset_index().sort_values(['stkcd','trd_dt','report_period'])
    # 如果在相同的trd_dt有不同的report_period记录，取report_period较大的那条记录
    df=df[~df.duplicated(['stkcd','trd_dt'],keep='last')]
    s=df.set_index(['stkcd','trd_dt']).sort_index()['result'].dropna()

    return s


#------------------------------------base function-----------------------------
'''
input must be DataFrame contained 'trd_dt',since we will set 'trd_dt' as index in 
the function '_adjust_result_format'.

the output is a series without name,and the index is ['stkcd','report_period']
'''

def raw_level(df, col, ttm=True):
    '''
    计算某个level的ttm
    df 是按季度的数据
    '''
    df['x']=df[col]
    if ttm:
        df['x']=ttm_adjust(df.x)
    df['result']=df['x']
    return _adjust_result_format(df)

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
    df['x']=df[col]
    if delete_negative:
        df['x'][df.x<=0]=np.nan
    if ttm:
        df.x=ttm_adjust(df.x)
    df['result']=df.x.groupby('stkcd').apply(lambda s:s.pct_change(periods=q))
    return _adjust_result_format(df)

def x_history_growth_avg(df,col,q=12,ttm=False,delete_negative=True):
    df['x']=df[col]
    if delete_negative:
        df['x'][df.x<=0]=np.nan
    if ttm:
        df.x=ttm_adjust(df.x)

    def cal(s,q):
        pct_chg=s.pct_change()
        return pct_chg.rolling(q,min_periods=int(q/2)).mean()

    df['result']=df.x.groupby('stkcd').apply(cal,q)
    return _adjust_result_format(df)

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
    df['x']=df[col]
    if delete_negative:
        df['x'][df.x<=0]=np.nan
    if ttm:
        df.x=ttm_adjust(df.x)

    def _cal_cumulative_g(arr):
        return np.cumprod((np.diff(arr)/arr[:-1])+1)[-1]-1

    df['result']=df.x.groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).apply(_cal_cumulative_g))
    return _adjust_result_format(df)

def x_history_std(df,col,q=8,ttm=True):
    '''
    std(x,q)
    Args:
        df:
        col:
        q: int,quarters
        ttm: boolean

    Returns: pd.Series

    '''
    df['x']=df[col]
    if ttm:
        df.x=ttm_adjust(df.x)
    df['result']=df.x.groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).std())
    return _adjust_result_format(df)

def x_history_downside_std(df,col,q=8,ttm=False):
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

    df['x']=df[col]
    if ttm:
        df.x=ttm_adjust(df.x)
    df['result']=df.x.groupby('stkcd').apply(downside_risk,q)
    return _adjust_result_format(df)

def ratio_x_y(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    x/y
    financial ratio in x/y
    '''
    df['x']=df[col1]
    df['y']=df[col2]
    if delete_negative_y:
        df['y'][df.y<=0]=np.nan

    if ttm:
        df.x=ttm_adjust(df.x)
        df.y=ttm_adjust(df.y)
    df['result']=df.x/df.y
    return _adjust_result_format(df)

def ratio_yoy_chg(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x/y)
    year-to-year change in financial ratio
    '''
    df['x']=df[col1]
    df['y']=df[col2]

    if delete_negative_y:
        df['y'][df.y<=0]=np.nan

    if ttm:
        df.x=ttm_adjust(df.x)
        df.y=ttm_adjust(df.y)
    df['ratio']=df.x/df.y
    df['result']=df['ratio'].groupby('stkcd').apply(
        lambda s:s-s.shift(4))
    return _adjust_result_format(df)

def ratio_yoy_pct_chg(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x/y)/(x/y)
    year-to-year "percent" change in financial ratio
    '''
    df['x']=df[col1]
    df['y']=df[col2]

    if delete_negative_y:
        df['y'][df.y<0]=np.nan

    if ttm:
        df.x=ttm_adjust(df.x)
        df.y=ttm_adjust(df.y)
    df['ratio']= df.x/df.y
    df['result']=df['ratio'].groupby('stkcd').apply(
        lambda s:s.pct_change(periods=4))
    return _adjust_result_format(df)

def pct_chg_dif(df, col1, col2, ttm=True,delete_negative=True):
    '''
    d(x)/x -d(y)/y
    the difference between the percentage change in each accounting variable and
    the percentage change in a base variable
    '''
    df['x']=df[col1]
    df['y']=df[col2]

    if delete_negative:
        df['x'][df.x<=0]=np.nan
        df['y'][df.y<=0]=np.nan
    if ttm:
        df.x=ttm_adjust(df.x)
        df.y=ttm_adjust(df.y)

    df['pct_chg_x']=df.x.groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['pct_chg_y']=df.y.groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['result']=df['pct_chg_x']-df['pct_chg_y']
    return _adjust_result_format(df)

def ratio_x_chg_over_lag_y(df, col1, col2, ttm=True,delete_negative_y=True):
    '''
    d(x)/lag(y)
    the change in each accounting variable scaled by a lagged base variable
    '''
    df['x']=df[col1]
    df['y']=df[col2]
    if delete_negative_y:
        df['y'][df.y<=0]=np.nan

    if ttm:
        df.x=ttm_adjust(df.x)
        df.y=ttm_adjust(df.y)
    df['x_chg']=df.x.groupby('stkcd').apply(lambda s: s - s.shift(1))
    df['lag_y']=df.y.groupby('stkcd').shift(1)
    df['result']=df['x_chg']/df['lag_y']
    return _adjust_result_format(df)




