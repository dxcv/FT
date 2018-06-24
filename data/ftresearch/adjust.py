# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-15  09:30
# NAME:FT-adjust.py
from datetime import timedelta

import pymysql
import pandas as pd
import os

from config import D_FT_ADJ, D_FT_RAW
from data.dataApi import read_from_sql
from tools import number2dateStr
import numpy as np


def create_cache():
    tbnames = [
        'equity_cash_dividend',
        'equity_consensus_forecast',
        'equity_fundamental_info',
        'equity_selected_balance_sheet',
        'equity_selected_cashflow_sheet',
        'equity_selected_cashflow_sheet_q',
        'equity_selected_income_sheet',
        'equity_selected_income_sheet_q',
        'equity_selected_indice_ir',
        'equity_selected_trading_data',
        'equity_shareholder_big10',
        'equity_shareholder_float_big10',
        'equity_shareholder_number',
    ]
    #TODO: unify the trd_dt of different tables
    #TODO: calculate my own q sheets

    for tbname in tbnames:
        print(tbname)
        df=read_from_sql(tbname,database='ftresearch')
        df.to_pickle(os.path.join(D_FT_RAW,tbname+'.pkl'))

def read_cache(tbname):
    df=pd.read_pickle(os.path.join(D_FT_RAW,tbname+'.pkl'))
    return df

def adjust_three_sheets():
    '''
    问题：
    1. equity_selected_income_sheet_q 中report_period 的顺序不对，那么直接rolling(4).sum()的时候使用的是错误的数据。
    例如，000001.SZ，report_period=2006-03-31的公布日期为2007-04-26 和report_period=2007-03-31 公布的日期一样。这
    是因为取得数据是“调整”数据吗？

    这种情况，在某个时间点，如果有调整的数据就是用调整后得数据，否则使用调整前得数据。
    比如，对于2016年的三季报，我们在2007-04-26前，我们应该使用调整前的数据，在此之后，
    应该使用调整后的数据，比如在2007-05的时候，我们应该使用2007-04-26 调整后的数据。
    '''
    tbnames=[
        'equity_selected_balance_sheet',
        'equity_selected_cashflow_sheet',
        'equity_selected_cashflow_sheet_q',
        'equity_selected_income_sheet',
        'equity_selected_income_sheet_q',
    ]

    #TODO: unify the trd_dt of different tables
    #TODO: calculate my own q sheets

    for tbname in tbnames:
        print(tbname)
        df=read_cache(tbname)
        date_cols=['trd_dt','ann_dt','report_period']
        for dc in date_cols:
            if dc in df.columns:
                df[dc]=pd.to_datetime(df[dc].map(number2dateStr))

        df=df.sort_values(['stkcd','report_period','trd_dt'])
        df=df[~df.duplicated(subset=['stkcd','report_period'],keep='last')]#trick:keep the latest item
        df=df.groupby('stkcd').apply(
            lambda x:x.set_index('report_period').resample('Q').asfreq()
        )

        df=df.drop(['stkcd','create_time','update_time'],axis=1)
        df.to_pickle(os.path.join(D_FT_ADJ, tbname + '.pkl'))

adjust_three_sheets()

def adjust_equity_cash_dividend():
    tbname='equity_cash_dividend'
    df=read_cache(tbname)
    date_cols = ['trd_dt', 'ann_dt', 'report_period']
    for dc in date_cols:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc].map(number2dateStr))

    df = df.sort_values(['stkcd', 'report_period', 'trd_dt'])
    df = df[~df.duplicated(subset=['stkcd', 'report_period'], keep='first')]
    df=df.set_index(['stkcd','report_period'])

    df = df.drop(['create_time', 'update_time'], axis=1)
    df.to_pickle(os.path.join(D_FT_ADJ, tbname + '.pkl'))

def adjust_equity_fundamental_info():
    tbname='equity_fundamental_info'
    df=read_cache(tbname)
    date_cols=['trd_dt','listdate']
    for dc in date_cols:
        if dc in df.columns:
            df[dc]=pd.to_datetime(df[dc].map(number2dateStr))
    # df=df.set_index(['stkcd','trd_dt']).sort_index()
    # adjust type_st,True means st,
    df['type_st']=df['type_st'].replace(1.0,True)
    df['type_st']=df['type_st'].fillna(False)

    df['young_1year']=np.where(df.trd_dt <= df.listdate + timedelta(days=365), True,False)
    df=df.set_index(['stkcd','trd_dt']).sort_index()
    df=df.drop(['create_time','update_time'],axis=1)
    df.to_pickle(os.path.join(D_FT_ADJ, tbname + '.pkl'))

def adjust_equity_selected_trading_data():
    tbname='equity_selected_trading_data'
    df=read_cache(tbname)
    df['trd_dt']=pd.to_datetime(df['trd_dt'].map(str))
    df=df.set_index(['stkcd','trd_dt'])
    df=df.sort_index()
    df.to_pickle(os.path.join(D_FT_ADJ,tbname+'.pkl'))

def adjust_equity_selected_indice_ir():
    tbname='equity_selected_indice_ir'
    df=read_cache(tbname)
    df['trd_dt']=pd.to_datetime(df['trd_dt'].map(str))
    df['sz50_ret_d']=df['sz50'].pct_change()
    df['hs300_ret_d']=df['hs300'].pct_change()
    df['zz500_ret_d']=df['zz500'].pct_change()
    df=df.set_index('trd_dt')
    df.to_pickle(os.path.join(D_FT_ADJ,tbname+'.pkl'))

# adjust_equity_selected_indice_ir()





# if __name__ == '__main__':
    # adjust_three_sheets()
    # adjust_equity_cash_dividend()
    # adjust_equity_fundamental_info()
    # adjust_equity_selected_trading_data()

