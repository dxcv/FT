# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-11  15:29
# NAME:FT_hp-test_roe.py
import multiprocessing
import pickle
from math import floor, ceil, sqrt

from data.dataApi import read_local
import pandas as pd
import os
from config import SINGLE_D_INDICATOR, SINGLE_D_CHECK, DCC, \
    FORWARD_TRADING_DAY, LEAST_CROSS_SAMPLE, DIR_SINGLE_BACKTEST
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tools import monitor, filter_st_and_young, clean

G = 10




def get_signal():
    name = 'Q__roe'
    trading = read_local('equity_selected_trading_data')
    df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
    df = df.stack().to_frame().swaplevel().sort_index()
    df.columns = [name]
    fdmt = read_local('equity_fundamental_info')
    data = pd.concat([fdmt, df,trading['adjclose']], axis=1, join='inner')
    # data=data[-int(data.shape[0]/10):]

    data = data.dropna(subset=['type_st', 'young_1year'])
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    data=data[(data['adjclose']<data['adjuplimit']) & (data['adjclose']>data['adjdownlimit'])]

    data = data.dropna(subset=['wind_indcd', name])
    # data = data.groupby('trd_dt').filter(lambda x: x.shape[0] > LEAST_CROSS_SAMPLE)

    cleaned_data = clean(data, col=name, by='trd_dt')
    signal = pd.pivot_table(cleaned_data, values=name, index='trd_dt',
                            columns='stkcd').sort_index()
    signal = signal.shift(1)  # trick:

    signal.to_pickle(r'E:\FT_Users\HTZhang\tmp\signal.pkl')
    print('signal finished!')

def test_daily():
    signal=pd.read_pickle(r'e:\FT_Users\HTZhang\tmp\signal.pkl')
    trading=read_local('equity_selected_trading_data')
    ret=pd.pivot_table(trading,values='pctchange',index='trd_dt',columns='stkcd')/100
    index=[ind for ind in signal.index if ind in ret.index]
    column=[col for col in signal.columns if col in ret.columns]
    signal=signal.reindex(index=index,columns=column)
    target=signal.apply(lambda s:s.nlargest(100),axis=1)
    mould=ret.copy().reindex(index=target.index,columns=target.columns)
    mould=mould[target.notnull()]
    mould.to_csv(r'e:\FT_Users\HTZhang\tmp\mould.csv')
    print('mould finished')
    avg=mould.mean(axis=1)
    zz500=read_local('equity_selected_indice_ir')['zz500_ret_d']
    comb=pd.concat([avg,zz500],axis=1,keys=['strategy','zz500'])
    comb=comb.dropna()
    comb['relative']=comb['strategy']-comb['zz500']
    comb.to_csv(r'e:\FT_Users\HTZhang\tmp\comb.csv')




    comb=pd.read_csv(r'e:\FT_Users\HTZhang\tmp\comb.csv',index_col=0,parse_dates=True)
    import matplotlib.pyplot as plt
    (1+comb).cumprod()['relative'].plot()
    plt.show()

signal_d=pd.read_pickle(r'e:\FT_Users\HTZhang\tmp\signal.pkl')
trading=read_local('equity_selected_trading_data')

signal_m=signal_d.resample('M').last()

# dates=[]
# stocks=[]
# for date,row in signal_m.iterrows():
#     stocks.append(row.nlargest(100))
#     dates.append(date)
#     print(date)
# ss=pd.concat(stocks,keys=dates)
#
# ss=ss.reset_index()
#
# ss.columns=['trd_dt','stkcd','o']
# ss=ss[['trd_dt','stkcd']].set_index('trd_dt',drop=True)
#
# ss.to_csv(r'e:\FT_Users\HTZhang\tmp\100.csv')

close_d=pd.pivot_table(trading,values='adjclose',index='trd_dt',columns='stkcd')
close_m=close_d.resample('M').last()

ret_m=close_m.pct_change()
ret_1m=ret_m.shift(-1)

index=[ind for ind in signal_m.index if ind in ret_1m.index]
column=[col for col in signal_m.columns if  col in ret_1m.columns]
signal_m=signal_d.reindex(index=index, columns=column)
target=signal_m.apply(lambda s:s.nlargest(100),axis=1)

mould=ret_1m.copy().reindex(index=target.index,columns=target.columns)
mould=mould[target.notnull()]

ir=read_local('equity_selected_indice_ir')
zz500_m=ir['zz500'].resample('M').last()
ret_zz500_1m=zz500_m.pct_change().shift(-1)

avg=mould.mean(axis=1)
comb=pd.concat([avg,ret_zz500_1m],axis=1,keys=['strategy','zz500'])
comb=comb.dropna()
comb['relative'] = comb['strategy'] - comb['zz500']

# comb.to_csv(r'e:\FT_Users\HTZhang\tmp\comb_monthly.csv')

# import matplotlib.pyplot as plt
# (1 + comb).cumprod()['relative'].plot()
# plt.show()

(comb['2013':'2016']+1).cumprod()['relative'].plot().get_figure().show()



