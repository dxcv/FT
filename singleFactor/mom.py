# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-24  22:16
# NAME:FT-mom.py
from config import SINGLE_D_INDICATOR_TECHNICAL
import os
import pandas as pd

from data.dataApi import read_local


def _save(df,name):
    new_name='T__'+name
    df.columns=[new_name]
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR_TECHNICAL,new_name+'.pkl'))

#=================================momentum=====================================
def _cal_mom(x,window):
    return x.resample('M',on='month_end')['close'].last().pct_change(periods=window)

def get_retn(n):
    '''
    get return of n month

    Args:
        n: window

    Returns:multiIndex Series,with the index as ['stkcd','trd_dt']

    '''
    trading_m=read_local('trading_m')
    return trading_m['close'].groupby('stkcd').pct_change(periods=n)


def get_moms():
    '''过去n个月的收益率'''
    windows=[1,2,3,6,9,12,24,36]
    for window in windows:
        name='mom_{}M'.format(window)
        mom=get_retn(window)
        mom=mom.to_frame()
        mom.columns=[name]
        _save(mom,name)
        print(window)

# get_moms()

def get_mom_mc():
    for l in [6,12]:
        name='mom_{}mc1m'.format(l)
        mom_l=get_retn(l)
        mom_s=get_retn(1)
        mom_mc=mom_l-mom_s
        mom_mc=mom_mc.to_frame()
        mom_mc.columns=[name]
        _save(mom_mc,name)

# get_mom_mc()

def get_mom_12mc6m():
    name='mom_12mc6m'
    mom_l=get_retn(12)
    mom_s=get_retn(6)
    mom_mc=mom_l-mom_s
    mom_mc=mom_mc.to_frame()
    mom_mc.columns=[name]
    _save(mom_mc,name)

# get_mom_12mc6m()
def get_mom_1dc1m():
    '''
    过去一个月最大日收益率
    Returns:

    '''
    name='mom_1dc1m'
    trading_d=read_local('equity_selected_trading_data')
    trading_d['ret_d']=trading_d['adjclose'].groupby('stkcd').pct_change()
    trading_d=trading_d.reset_index()
    mom_max_ret=trading_d.groupby('stkcd').apply(
        lambda x:x.resample('M',on='trd_dt',closed='right',label='right')
            .apply({'ret_d':'max','trd_dt':'last'}))
    mom_max_ret.index.names=['stkcd','month_end']
    mom_max_ret=mom_max_ret.reset_index().set_index(['stkcd','trd_dt'])
    del mom_max_ret['month_end']
    mom_max_ret.columns=[name]
    _save(mom_max_ret,name)

def get_sucess():
    #Success=1-过去一个月的收益率排名/股票总数
    #等价于反转因子(1M)
    name='sucess'
    trading_m=read_local('trading_m')

    def func(x):
        x=x.reset_index()
        x['rank']=x['ret_m'].rank()
        x['sucess1']=1-x['rank']/x.shape[0]
        x['sucess']=1-x['ret_m'].rank(ascending=True)/x.shape[0] # the larger the ret_m,the larger the rank and the smaller the sucess value
        x=x.set_index('stkcd')
        return x['sucess']

    trading_m=trading_m.swaplevel()
    sucess=trading_m.groupby('month_end').apply(func)
    #trick: get the last trading date of a calendar month
    month_end_trd_dt=trading_m.reset_index().sort_values(['month_end','trd_dt']).groupby('month_end')['trd_dt'].max()
    month_end_trd_dt.name='month_end_trd_dt'
    sucess=sucess.unstack()
    comb=pd.concat([sucess,month_end_trd_dt],axis=1).set_index('month_end_trd_dt')

    result=comb.T.stack().to_frame()
    result.index.names=['stkcd','trd_dt']
    result.columns=[name]
    _save(result,name)

def get_pm_1d():
    '''上一个交易日收益率'''
    name='pm_1d'
    trading_d=read_local('equity_selected_trading_data')
    trading_d[name]=trading_d['adjclose'].groupby('stkcd').pct_change()
    _save(trading_d[[name]],name)

if __name__ == '__main__':
    get_moms()
    get_mom_mc()
    get_mom_12mc6m()
    get_mom_1dc1m()
    # get_sucess()
    get_pm_1d()





