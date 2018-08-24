# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-12  14:33
# NAME:FT_hp-vol_and_beta_daily.py
import os

from config import SINGLE_D_INDICATOR
from data.dataApi import read_local
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
from tools import myroll


DAYS=[10, 20, 60, 120, 250]

trading = read_local('equity_selected_trading_data')
ret = pd.pivot_table(trading, values='pctchange', index='trd_dt',
                     columns='stkcd') / 100
zz500_ret_d = read_local('equity_selected_indice_ir')['zz500_ret_d']
df = pd.concat([zz500_ret_d, ret], axis=1, join='inner')
df = df.dropna(subset=['zz500_ret_d'])
df = df.dropna(how='all')

# df=df[-int(df.shape[0]/100):] #fixme

def save_indicator(df,name):
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))


def beta(df, d):
    df=df.dropna(thresh=int(d / 2), axis=1)
    df=df.fillna(df.mean()) #Trick: fillna with average
    # df=df.fillna(0)
    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    return pd.Series(b[1],index= df.columns[1:], name='beta')

def idioVol(df, d):
    df=df.dropna(thresh=int(d / 2), axis=1)
    df=df.fillna(df.mean())

    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:]) #beta
    resid=df.values[:,1:]-X.dot(b)# real value - fitted value
    resid_std=np.std(resid,axis=0)
    return pd.Series(resid_std,index=df.columns[1:],name='idiovol')

def cal_betas():
    #TODO: employ multiprocessing
    for d in [30,60,180,300]:
        name='T__beta_{}'.format(d)
        results=myroll(df, d).apply(beta, d)
        save_indicator(results.unstack(),name)
        print(d)

def cal_idioVol():
    #fixme: shape is (154,1223), missing value problem
    for d in [30,60,180,300]:
        name='T__idioVol_{}'.format(d)
        results=myroll(df, d).apply(idioVol, d)
        save_indicator(results.unstack(),name)
        print(d)

def get_high_minus_low():
    trading = read_local('equity_selected_trading_data')
    for day in DAYS:
        adjhigh=pd.pivot_table(trading,values='adjhigh',index='trd_dt',columns='stkcd')
        adjlow=pd.pivot_table(trading,values='adjlow',index='trd_dt',columns='stkcd')

        window_adjhigh=adjhigh.rolling(day,min_periods=int(day/2)).max()
        window_adjlow=adjlow.rolling(day,min_periods=int(day/2)).min()
        high_minus_low=window_adjhigh-window_adjlow
        name='T__vol_high_minus_low_{}'.format(day)
        save_indicator(high_minus_low,name)

def get_std():
    trading = read_local('equity_selected_trading_data')
    ret=pd.pivot_table(trading,values='pctchange',index='trd_dt',columns='stkcd')/100
    for day in DAYS:
        std=ret.rolling(day,min_periods=int(day/2)).std()
        name='T__vol_std_{}'.format(day)
        save_indicator(std,name)
        print(day)

def get_vol_amount():#TODO： std/mean    idiosyncratic
    trading = read_local('equity_selected_trading_data')
    for day in DAYS:
        amount=pd.pivot_table(trading,values='amount',index='trd_dt',columns='stkcd')
        amount_std=amount.rolling(day,min_periods=int(day/2)).std()
        amount_avg=amount.rolling(day,min_periods=int(day/2)).mean()
        vol_amount=amount_std/amount_avg
        name='T__vol_amount_{}'.format(day)
        save_indicator(vol_amount,name)
        print(day)

if __name__ == '__main__':
    cal_betas()
    cal_idioVol()
    get_high_minus_low()
    get_std()
    get_vol_amount()
