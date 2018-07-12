# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-06  17:53
# NAME:FT-a.py

import pandas as pd
from backtest.main import quick
import numpy as np
from data.dataApi import read_local

trading = read_local('equity_selected_trading_data')
ret=pd.pivot_table(trading,values='pctchange',index='trd_dt',columns='stkcd')/100
zz500_ret_d=read_local('equity_selected_indice_ir')['zz500_ret_d']
df=pd.concat([zz500_ret_d,ret],axis=1,join='inner')
df=df.dropna(how='all')

def roll(df, d):
    # stack df.values d-times shifted once at each stack
    roll_array = np.dstack([df.values[i:i + d, :] for i in range(len(df.index) - d + 1)]).T
    # roll_array is now a 3-D array and can be read into
    # a pandas panel object
    panel = pd.Panel(roll_array,
                     items=df.index[d - 1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(d), name='roll'))
    # convert to dataframe and pivot + groupby
    # is now ready for any action normally performed
    # on a groupby object
    return panel.to_frame().unstack().T.groupby(level=0)


def beta(df, d):
    df=df.dropna(thresh=int(d / 2), axis=1)
    df=df.fillna(0)
    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    return pd.Series(b[1], df.columns[1:], name='Beta')



d=20
results=roll(df,d).apply(beta,d)



