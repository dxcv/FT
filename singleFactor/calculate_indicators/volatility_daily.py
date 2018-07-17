# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-12  11:06
# NAME:FT_hp-volatility_daily.py

import os

from config import SINGLE_D_INDICATOR
from data.dataApi import read_local
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np

def save_indicator(df,name):
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))


DAYS=[10, 20, 60, 120, 250]

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

def get_amount2std():
    trading = read_local('equity_selected_trading_data')
    for day in DAYS:
        amount=pd.pivot_table(trading,values='amount',index='trd_dt',columns='stkcd')
        adjclose=pd.pivot_table(trading,values='adjclose',index='trd_dt',columns='stkcd')
        amount_avg=amount.rolling(day,min_periods=int(day/2)).mean()
        adjclose_std=adjclose.rolling(day,min_periods=int(day/2)).std()
        amount_to_std=amount_avg/adjclose_std
        name='T__vol_amount2std_{}'.format(day)
        save_indicator(amount_to_std,name)
        print(day)

if __name__ == '__main__':
    get_amount2std()






