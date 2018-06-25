# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  09:02
# NAME:FT-new_operators.py
import numpy as np

def x_ttm(s,q=4):
    return s.groupby('stkcd').rolling(q).mean()

def x_chg(s,q=1):
    '''
    d(x)
    Args:
        s:
        q:

    Returns:

    '''
    return s.groupby('stkcd').apply(lambda x:x-x.shift(q))

def x_pct_chg(s,q=1,delete_negative=True):
    '''
    d(x)/x
    Args:
        s:按季度的数据
        q:int,季度数，如果是求同比增长，则取q=4
        delete_negative:

    Returns:

    '''
    if delete_negative:
        s=s.where(s>0,np.nan)
    return s.groupby('stkcd').apply(lambda x:x.pct_change(periods=q))

def x_history_growth_avg(s, q=12, delete_negative=True):
    if delete_negative:
        s=s.where(s>0,np.nan)

    def _cal(_s, q):
        pct_chg=_s.pct_change()
        return pct_chg.rolling(q,min_periods=int(q/2)).mean()
    return s.groupby('stkcd').apply(_cal,q)

def x_square(s):
    return s*s

def x_history_compound_growth(s,q=20,delete_negative=True):
    '''
        计算过去q个季度的复合增长率
    Args:
        s:
        q:
        delete_negative_True:

    Returns:

    '''
    if delete_negative:
        s=s.where(s>0,np.nan)

    def _cal_cumulative_g(arr):
        return np.cumprod((np.diff(arr)/arr[:-1])+1)[-1]-1

    return s.groupby('stkcd').apply(
        lambda x:x.rolling(q,min_periods=q).apply(_cal_cumulative_g))

def x_history_std(s, q=8):
    '''
    std(x,q)
    '''
    #TODO：市值处理？
    return s.groupby('stkcd').apply(
        lambda x:x.rolling(q,min_periods=q).std())

def x_history_downside_std(s,q=8):
    '''
    stddev(min(x-x(-1),0))

    #TODO:normalize (scale with the mean value)
    '''
    def downside_risk(_s, q):
        dev = _s - _s.shift(1)
        downside = dev.where(dev < 0, 0)
        r = downside.rolling(q, min_periods=q).std()
        return r
    return s.groupby('stkcd').apply(downside_risk,q)

def x_history_growth_std(s,q=12,delete_negative=True):
    if delete_negative:
        s=s.where(s>0,np.nan)
    return x_history_std(x_pct_chg(s,q=1),q=q)

def x_history_growth_downside_std(s,q=12,delete_negative=True):
    if delete_negative:
        s=s.where(s>0,np.nan)
    return x_history_downside_std(x_pct_chg(s,q=1),q=q)

#----------------with two indicators-------------------------------------------
def ratio(df, x, y, delete_negative_y=True, smooth=False, handle_inf=True):
    '''
        x/y
    Args:
        df:
        x:
        y:
        delete_negative_y:
        smooth:True or False,if True,  x*2/(y+lag(y))
    Returns:

    '''
    if delete_negative_y:
        df[y]=df[y].where(df[y]>0,np.nan)
    if smooth:
        y_smooth=df[y].groupby('stkcd').apply(lambda s:(s+s.shift(1))/2)
        ratio=df[x]/y_smooth
    else:
        ratio=df[x]/df[y]
    if handle_inf:
        ratio=ratio.where((-np.inf<ratio)&(ratio<np.inf),np.nan)#TODO: 注意 除法 容易出现inf 和 -inf
    return ratio

def ratio_chg(df, x, y, q=4, delete_negative_y=True):
    '''
        d(x/y)
    Args:
        df:
        x:
        y:
        delete_negative_y:

    Returns:
    '''
    return x_chg(ratio(df, x, y, delete_negative_y), q=q)

def ratio_pct_chg(df, x, y, q=4, delete_negative_y=True):
    '''
        d(x/y)/(x/y)
    '''
    return x_pct_chg(ratio(df, x, y, delete_negative_y), q=q)

def ratio_history_std(df,x,y,q=8,delete_negative_y=True):
    return x_history_std(ratio(df, x, y, delete_negative_y), q=q)

def ratio_history_compound_growth(df,x,y,q=8,delete_negative_y=True):
    return x_history_compound_growth(ratio(df,x,y,delete_negative_y),q=q)

def ratio_history_downside_std(df,x,y,q=8,delete_negative_y=True):
    return x_history_downside_std(ratio(df, x, y, delete_negative_y), q=q)

def pct_chg_dif(df,x,y,q=1,delete_negative=True):
    '''
        d(x)/x-d(y)/y
    Args:
        df:
        x:
        y:
        delete_negative:

    Returns:

    '''
    if delete_negative:
        df[x]=df[x].where(df[x]>0,np.nan)
        df[y]=df[y].where(df[y]>0,np.nan)
    return x_pct_chg(df[x],q)-x_pct_chg(df[y],q)

def ratio_x_chg_over_lag_y(df, x,y,delete_negative_y=True):
    '''
    d(x)/lag(y)
    the change in each accounting variable scaled by a lagged base variable
    '''
    if delete_negative_y:
        df[y] = df[y].where(df[y] > 0, np.nan)

    chg_x=x_chg(df[x],q=1)
    lag_y=df[y].groupby('stkcd').shift(1)
    return chg_x/lag_y


