# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-23  20:53
# NAME:FT-a.py
import os
import pandas as pd

from config import FORWARD_TRADING_DAY
from data.dataApi import read_local

G=10

def get_cover_rate():
    fdmt = read_local('equity_fundamental_info')

    base=pd.pivot_table(fdmt,values='cap',index='trd_dt',columns='stkcd')
    base_monthly=base.resample('M').last()

    col=r'G_hcg_12__s_fa_roe'
    df=pd.read_pickle(r'D:\zht\database\quantDb\internship\FT\singleFactor\indicators\G_hcg_12__s_fa_roe.pkl')
    df=df.reset_index()
    table=pd.pivot_table(df,values=col,index='trd_dt',columns='stkcd')
    table=table.reindex(base.index)
    table=table.ffill(limit=400)
    monthly=table.resample('M').last()

    total=base_monthly.notnull().sum(axis=1)
    covered=monthly.notnull().sum(axis=1)
    cover_rate=covered/total
    cover_rate.plot().get_figure().show()

fdmt = read_local('equity_fundamental_info')

col = r'G_hcg_12__s_fa_roe'
df = pd.read_pickle(
    r'D:\zht\database\quantDb\internship\FT\singleFactor\indicators\G_hcg_12__s_fa_roe.pkl')

data=pd.merge(fdmt.reset_index(),df.reset_index(),on=['stkcd','trd_dt'],how='left')
data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
data = data.groupby('stkcd').ffill(limit=FORWARD_TRADING_DAY) # review: 向前填充最最多400个交易日

df=data[['stkcd','trd_dt','cap',col]]
monthly=df.groupby('stkcd').resample('M',on='trd_dt').last()
monthly.index.names=['stkcd','month_end']
monthly['g'] = monthly.groupby('month_end', group_keys=False).apply(
    lambda x: pd.qcut(x['cap'], G,
                      labels=['g{}'.format(i) for i in range(1, G + 1)]))


cover_rate=monthly.groupby(['month_end', 'g']).apply(lambda x: x[col].notnull().sum() / x.shape[0])
cover_rate= cover_rate.unstack('g') / G
cover_rate=cover_rate[['g{}'.format(i) for i in range(1, G + 1)]]



import matplotlib.pyplot as plt

plt.stackplot(cover_rate.index, cover_rate.T.values, alpha=0.7)
plt.show()




