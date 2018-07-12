# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-12  17:02
# NAME:FT_hp-check_by_panel.py
import os
import pandas as pd
from backtest.main import quick
from config import SINGLE_D_INDICATOR, DIR_BACKTEST_RESULT, LEAST_CROSS_SAMPLE
from data.dataApi import read_local
from singleFactor.check import daily_to_monthly, check_factor
from tools import clean
import numpy as np

'''
数据统一使用 一维的dataframe，每张表存储一个指标，便于向量化处理，如果用MultiIndex的话
groupby 其实是很慢的。

'''


name = 'C__est_bookvalue_FT24M_to_close_g_20'


df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
df = df.stack().to_frame().swaplevel().sort_index()
df.columns = [name]
fdmt = read_local('equity_fundamental_info')
data = pd.concat([fdmt, df], axis=1, join='inner')

data = data.dropna(subset=['type_st', 'young_1year'])
data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
data = data.dropna(subset=['wind_indcd', name])
data = data.groupby('trd_dt').filter(lambda x: x.shape[0] > LEAST_CROSS_SAMPLE)



daily=pd.pivot_table(data,values=name,index='trd_dt',columns='stkcd')
monthly=daily.resample('M').last()

monthly.to_pickle(r'E:\FT_Users\HTZhang\tmp\monthly.pkl')


#TODO:cover rate

def handle_outlier(x, k=4.5):
    '''
    Args:
        x:array,without NaN
        k:
    Returns:array

    '''
    med=np.median(x)
    mad = np.median(np.abs(x - med))
    uplimit = med + k * mad
    lwlimit = med - k * mad
    y = np.where(x >= uplimit, uplimit, np.where(x <= lwlimit, lwlimit, x))
    return y

def z_score(x):
    return (x-np.mean(x))/np.std(x)

monthly=pd.read_pickle(r'E:\FT_Users\HTZhang\tmp\monthly.pkl')
m1=monthly.stack().swaplevel().sort_index().to_frame()
m1.columns=[name]


s=monthly.iloc[0,:]
s=s.dropna()
s1=handle_outlier(s.values)













