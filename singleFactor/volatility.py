# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-28  14:49
# NAME:FT-volatility.py
from data.dataApi import read_local
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np


trading=read_local('equity_selected_trading_data')
trading=trading[-int(trading.shape[0]/100):]#debug

window='1M'

# trd_dt=trading.groupby('stkcd').resample(window,on='trd_dt').last()['trd_dt']


dict_D={'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450}


def highlow(x):
    return x['adjhigh'].max()/x['adjlow'].min()

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

def groupby_rolling(x, dict, type_func):
    values = []
    #TODO:why not use map or other higher-order function

    names=[]
    for history, thresh in dict.items():
        # days=x['trd_dt'].sort_values().unique()
        days = sorted(x.index.get_level_values('trd_dt').unique())
        # or (days+MonthEnd(0)).unique()
        months=pd.date_range(start=days[0],end=days[-1],freq='M')
        value = x.groupby('stkcd').apply(
            lambda df: _rolling_for_series(df, months, history, thresh, type_func))
        values.append(value.T)
        names.append(history)
        print(history,thresh)
    result = pd.concat(values, axis=0,keys=names)
    return result

test=groupby_rolling(trading,dict_D,highlow)

test.to_csv(r'e:\a\test.csv')
print('a')

# std_1m=trading.groupby('stkcd').resample(window,on='trd_dt').apply(lambda x:x['adjclose'].pct_change().std())
# v2std_1m=trading.groupby('stkcd').resample(window,on='trd_dt').apply(
#     lambda x:x['amount'].sum()/(x['adjclose'].pct_change().std()))

# history=180





# indice=read_local('equity_selected_indice_ir')
# comb=trading.set_index(['stkcd','trd_dt']).join(indice['hs300_ret_d'])
# capm_residual=comb.groupby('stkcd').apply(
#     lambda x:x.rolling(history).apply(lambda xx:)
# )

