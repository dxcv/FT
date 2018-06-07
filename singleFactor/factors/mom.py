# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-04  22:55
# NAME:FT-mom.py


from data.dataApi import read_local_pkl
from singleFactor.factors.check import _check
import pandas as pd

#=================================momentum=====================================
def _cal_mom(x,window):
    x['mom_{}M'.format(window)]=x['close'].pct_change(periods=window)
    return x[['mom_{}M'.format(window),'trd_dt']]

def get_moms():
    '''过去n个月的收益率'''
    windows=[1,2,3,6,9,12,24,36]
    trading_m=read_local_pkl('trading_m')
    for window in windows:
        mom=trading_m.groupby('stkcd').apply(_cal_mom,window)
        mom=mom.reset_index().set_index(['stkcd','trd_dt'])
        _check(mom[['mom_{}M'.format(window)]],'mom_{}M'.format(window))

def get_mom_mc():
    '''过去n个月的收益率-过去1个月的收益率'''
    for l in [6,12]:
        name='mom_{}mc1m'.format(l)
        trading_m=read_local_pkl('trading_m')
        mom_l=trading_m.groupby('stkcd').apply(_cal_mom, l)
        mom_s=trading_m.groupby('stkcd').apply(_cal_mom, 1)
        df=pd.concat([mom_l[['trd_dt', 'mom_{}M'.format(l)]], mom_s['mom_1M']], axis=1)
        df[name]=df['mom_{}M'.format(l)]-df['mom_1M']
        df=df.reset_index().set_index(['stkcd','trd_dt'])
        _check(df[[name]],name)

def get_mom_12mc6m():
    trading_m=read_local_pkl('trading_m')
    mom12=trading_m.groupby('stkcd').apply(_cal_mom,12)
    mom6=trading_m.groupby('stkcd').apply(_cal_mom,6)
    df=pd.concat([mom12[['trd_dt','mom_12M']],mom6['mom_6M']],axis=1)
    df['12mc6m']=(1+df['mom_12M'])/(1+df['mom_6M'])-1
    df=df.reset_index().set_index(['stkcd','trd_dt'])
    _check(df[['12mc6m']],'mom_12mc6m')

def get_mom_1dc1m():
    trading_d=read_local_pkl('equity_selected_trading_data')
    trading_d['ret_d']=trading_d['adjclose'].groupby('stkcd').pct_change()
    trading_d=trading_d.reset_index()
    mom_max_ret=trading_d.groupby('stkcd').apply(
        lambda x:x.resample('M',on='trd_dt',closed='right',label='right')
            .apply({'ret_d':'max','trd_dt':'last'}))
    mom_max_ret.index.names=['stkcd','month_end']
    mom_max_ret=mom_max_ret.reset_index().set_index(['stkcd','trd_dt'])
    _check(mom_max_ret[['ret_d']],'mom_1dc1m')

#===================================technical===================================
def get_sucess():
    #Success=1-过去一个月的收益率排名/股票总数
    name='T_sucess'
    trading_m=read_local_pkl('trading_m')

    def func(x):
        x=x.reset_index()
        x['sucess']=1-x['ret_m'].rank()/x.shape[0]
        x=x.set_index('stkcd')
        return x['sucess']

    trading_m=trading_m.swaplevel()
    trading_m[name]=trading_m.groupby(level='month_end').apply(func)
    trading_m=trading_m.reset_index().set_index(['stkcd','trd_dt'])
    _check(trading_m[[name]],name)

def get_pm_1d():
    #过去 1 日收益率
    name='T_pm_1d'
    trading_d=read_local_pkl('equity_selected_trading_data')
    trading_d[name]=trading_d['adjclose'].groupby('stkcd').pct_change()
    _check(trading_d[[name]],name)


if __name__ == '__main__':
    get_moms()
    get_mom_mc()
    get_mom_12mc6m()
    get_mom_1dc1m()
    get_sucess()
    get_pm_1d()