# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-04  16:21
# NAME:FT-calculate_derivetives.py
from data.dataApi import read_local_pkl
from config import DCSV,DPKL
import os

def _save(df,name):
    df.to_csv(os.path.join(DCSV,name+'.csv'))
    df.to_pickle(os.path.join(DPKL,name+'.pkl'))

def daily2monthly(x):
    ohlc_dict={
                # 'preclose':'last',
               # 'open':'first',
               # 'high':'max',
               # 'low':'min',
               # 'close':'last',
               'volume':'sum',
               # 'amount':'sum',
               # 'adjpreclose':'last',
               'adjopen':'first',
               'adjhigh':'max',
               'adjlow':'min',
               'adjclose':'last',
               # 'avgprice':'mean',
               'trd_dt':'last'
               }
    monthly=x.resample('M',on='trd_dt',closed='right',label='right').apply(ohlc_dict)
    # monthly=x.resample('M',how=ohlc_dict,closed='right',label='right',on='trd_dt')
    monthly['ret_m']=monthly['adjclose'].pct_change()
    '''
    asfreq('M') 这一行可以省略，因为daily的数据没有缺失交易日
    '''
    monthly=monthly.asfreq('M') #span
    return monthly

def get_monthly_trading_data():
    daily=read_local_pkl('equity_selected_trading_data')
    daily=daily.reset_index()
    monthly=daily.groupby('stkcd').apply(daily2monthly)
    monthly=monthly.rename(columns={'adjopen':'open',
                            'adjhigh':'high',
                            'adjlow':'low',
                            'adjclose':'close',
                            })
    monthly=monthly[['trd_dt']+[col for col in monthly.columns if col!='trd_dt']]
    monthly.index.names=['stkcd','month_end']
    _save(monthly,'trading_m')

# 原始数据中在停牌的时候股价是向前填充，ohlc 均为上一个交易日的收盘价，交易量为0

#TODO: use this method to handle quarterly data

# get_monthly_trading_data()


