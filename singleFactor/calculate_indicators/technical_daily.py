# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-12  09:36
# NAME:FT_hp-technical_daily.py
from config import SINGLE_D_INDICATOR
from data.dataApi import read_local
import pandas as pd
import os

def save_indicator(df,name):
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))



def get_mom():
    '''the return of the last N days'''
    trading=read_local('equity_selected_trading_data')
    adjclose=pd.pivot_table(trading,values='adjclose',index='trd_dt',columns='stkcd')
    days=[1,10,20,30,50,100,180,300]
    for day in days:
        name='T__mom_{}'.format(day)
        g=adjclose.pct_change(periods=day)
        save_indicator(g,name)
        print(day)


def get_mom_mc():
    '''the return of the last L days minus the return of last S days'''
    trading=read_local('equity_selected_trading_data')
    adjclose=pd.pivot_table(trading,values='adjclose',index='trd_dt',columns='stkcd')

    ls=[120,250]
    ss=[10,20]
    for l in ls:
        for s in ss:
            g=adjclose.pct_change(periods=l)-adjclose.pct_change(periods=s)
            name='T__mom_{}minus{}'.format(l,s)
            save_indicator(g,name)

def get_mom_maxRet_20():
    day=20
    name='T__mom_maxRet_{}'.format(day)
    trading=read_local('equity_selected_trading_data')
    adjclose=pd.pivot_table(trading,values='adjclose',index='trd_dt',columns='stkcd')
    ret=adjclose.pct_change()
    g=ret.rolling(day).max()
    save_indicator(g,name)





