# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-28  14:49
# NAME:FT-4volatility.py
import os

from config import SINGLE_D_INDICATOR
from data.dataApi import read_local
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np


trading=read_local('equity_selected_trading_data')
hs300_ret_d=read_local('equity_selected_indice_ir')['hs300_ret_d']
trading=trading.join(hs300_ret_d)
trading['pctchange']/=100 #trick:adjust the unit
# trading=trading[-int(trading.shape[0]/100):]#debug


dict_D={'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450}

def _save(df):
    '''

    Args:
        df:DataFrame with only one column and the index is
        ['stkcd','monh_end']

    Returns:

    '''
    name='T__vol_{}'.format(df.columns[0])
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))

def highlow(x):
    '''

    Args:
        x:DataFrame,the index is 'trd_dt',and the columns contains
        ['stkcd','preclose','open'] and so on.

    Returns:float

    '''
    return x['adjhigh'].max()/x['adjlow'].min()

def ret_std(x):
    return (x['pctchange']/100).std()

def amountToPriceStd(x):
    return x['amount'].sum()/x['close'].std()

def beta(x):
    a=x[['hs300_ret_d']].values
    A = np.hstack([a, np.ones([len(a), 1])])
    y=x[['pctchange']].values
    beta=np.linalg.lstsq(A,y,rcond=None)[0][0]
    return beta

def idioVol_capm(x):
    a=x[['hs300_ret_d']].values
    A = np.hstack([a, np.ones([len(a), 1])])
    y=x[['pctchange']].values
    resid = np.linalg.lstsq(A, y, rcond=None)[0][1]
    return np.std(resid)


def _rolling_for_series(df,months,history,thresh,type_func):
    df=df.reset_index(level='stkcd')
    values=[]
    for month in months:
        subdf=df.loc[:month].last(history)
        df=df.dropna()
        if subdf.shape[0]>thresh:
            values.append(type_func(subdf))
        else:
            values.append(np.nan)
    return pd.Series(values,index=months)

def groupby_rolling(x, dict, func_name):
    '''

    Args:
        x:
        dict:
        func_name:str,the function name

    Returns:DataFrame,with multiple columns and the index is ['stkcd','month_end']

    '''
    values = []
    #TODO:why not use map or other higher-order function
    names=[]
    for history, thresh in dict.items():
        # days=x['trd_dt'].sort_values().unique()
        days = sorted(x.index.get_level_values('trd_dt').unique())
        # or (days+MonthEnd(0)).unique()
        months=pd.date_range(start=days[0],end=days[-1],freq='M')
        value = x.groupby('stkcd').apply(
            lambda df: _rolling_for_series(df, months, history, thresh, eval(func_name)))
        values.append(value.T)
        names.append('{}_{}'.format(func_name,history))
        print(history,thresh)
    result = pd.concat(values, axis=0,keys=names)
    result.index.names=['history','month_end']
    result=result.stack().unstack('history').swaplevel().sort_index()
    return result

def cal_all():
    funcs=['highlow','ret_std','amountToPriceStd','beta','idioVol_capm']
    for func in funcs:
        df=groupby_rolling(trading,dict_D,func)
        for col in df.columns:
            _save(df[[col]])
        print(func)

def debug():
    func='idioVol_capm'
    df=groupby_rolling(trading,dict_D,func)
    for col in df.columns:
        _save(df[[col]])


cal_all()

