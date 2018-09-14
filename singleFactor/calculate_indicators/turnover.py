# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-16  14:40
# NAME:FT_hp-turnover.py
import multiprocessing
import os
from functools import partial

import time
from config import SINGLE_D_INDICATOR
from data.dataApi import read_local
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
from tools import mytiming

DAYS=[10,20,30,60,120,180,300]

trading = read_local('equity_selected_trading_data')
fdmt=read_local('equity_fundamental_info')

comb=pd.concat([trading,fdmt],axis=1)
comb['turnover1']=comb['amount']/comb['freeshares']
comb['turnover2']=comb['amount']/comb['freefloat_cap']

def save_indicator(df,name):
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))


def avg():
    #average turnover of the past n days
    for _type in ['turnover1','turnover2']:
        for day in DAYS:
            name='T__{}_avg_{}'.format(_type,day)
            to=pd.pivot_table(comb,values=_type,index='trd_dt',columns='stkcd')
            signal=to.rolling(day,min_periods=int(day/2)).mean()
            save_indicator(signal,name)
            print(_type,name)

def std():
    for _type in ['turnover1','turnover2']:
        for day in DAYS:
            name='T__{}_std_{}'.format(_type,day)
            to=pd.pivot_table(comb,values=_type,index='trd_dt',columns='stkcd')
            signal=to.rolling(day,min_periods=int(day/2)).std()
            save_indicator(signal,name)
            print(_type,name)


def relative_avg():
    # average turnover of past S days / average turnover of past L days
    for L in [20,40,60,180,300]:
        S=int(L/2)
        for _type in ['turnover1','turnover2']:
            name='T__{}_relative_avg_{}_{}'.format(_type,L,S)
            to = pd.pivot_table(comb, values=_type, index='trd_dt', columns='stkcd')
            avgL = to.rolling(L, min_periods=int(L / 2)).mean()
            avgS = to.rolling(S, min_periods=int(S / 2)).mean()
            signal=avgL/avgS
            save_indicator(signal, name)
            print(_type, name)

def relative_std():
    # turnover std of past S days / turnover std of past L days
    for L in [20, 40, 60, 180, 300]:
        S = int(L/ 2)
        for _type in ['turnover1', 'turnover2']:
            name = 'T__{}_relative_std_{}_{}'.format(_type, L, S)
            to = pd.pivot_table(comb, values=_type, index='trd_dt',
                                columns='stkcd')
            avgL = to.rolling(L, min_periods=int(L / 2)).std()
            avgS = to.rolling(S, min_periods=int(S / 2)).std()
            signal = avgL / avgS
            save_indicator(signal, name)
            print(_type, name)


def corrRetTurnover():
    def _corrRetTurnover(df, t_type, d):
        corr_series = df[t_type].rolling(d, min_periods=int(d / 2)).corr(
            other=df['pctchange'])
        return corr_series

    for _type in ['turnover1','turnover2']:
        for day in DAYS:
            name='T__{}_corrRetTurnover_{}'.format(_type,day)
            signal=comb.groupby('stkcd',group_keys=False).apply(partial(_corrRetTurnover,t_type=_type,d=day)).unstack('stkcd')
            save_indicator(signal,name)
            print(_type,day)

def run_parallel():
    for fname in ['avg','std','relative_avg','relative_std','corrRetTurnover']:
        p=multiprocessing.Process(target=eval(fname))
        p.start()
        # p.join()


def main():
    run_parallel()

if __name__ == '__main__':
    main()#13 minutes




