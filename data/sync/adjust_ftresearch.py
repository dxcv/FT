# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-02  16:36
# NAME:FT-adjust_ftresearch.py

import os
import pandas as pd

from config import DCSV, DPKL
from data.dataApi import read_raw
from tools import number2dateStr


def adjust_ftresearch():
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
        df=read_raw(tbname).reset_index(drop=True)
        date_cols=['trd_dt','ann_dt','report_period']
        for dc in date_cols:
            if dc in df.columns:
                df[dc]=pd.to_datetime(df[dc].map(number2dateStr))

        df=df.sort_values(['stkcd','report_period','trd_dt'])
        df=df[~df.duplicated(subset=['stkcd','report_period'],keep='first')]
        df=df.groupby('stkcd').apply(
            lambda x:x.set_index('report_period').resample('Q').asfreq()
        )

        # df=df.drop(['stkcd','create_time','update_time'],axis=1)

        df.to_csv(os.path.join(DCSV,tbname+'.csv'))
        df.to_pickle(os.path.join(DPKL,tbname+'.pkl'))

def adjust_equity_cash_dividend():
    tbname='equity_cash_dividend'
    df=read_raw(tbname).reset_index(drop=True)
    date_cols = ['trd_dt', 'ann_dt', 'report_period']
    for dc in date_cols:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc].map(number2dateStr))

    df = df.sort_values(['stkcd', 'report_period', 'trd_dt'])
    df = df[~df.duplicated(subset=['stkcd', 'report_period'], keep='first')]
    df=df.set_index(['stkcd','report_period'])

    # df = df.drop(['create_time', 'update_time'], axis=1)
    df.to_csv(os.path.join(DCSV, tbname + '.csv'))
    df.to_pickle(os.path.join(DPKL, tbname + '.pkl'))

def adjust_equity_fundamental_info():
    tbname='equity_fundamental_info'
    df=read_raw(tbname)
    date_cols=['trd_dt','listdate']
    for dc in date_cols:
        if dc in df.columns:
            df[dc]=pd.to_datetime(df[dc].map(number2dateStr))
    df=df.set_index(['stkcd','trd_dt']).sort_index()
    df.to_csv(os.path.join(DCSV, tbname + '.csv'))
    df.to_pickle(os.path.join(DPKL, tbname + '.pkl'))

def adjust_equity_selected_trading_data():
    tbname='equity_selected_trading_data'
    df=read_raw(tbname)
    df['trd_dt']=pd.to_datetime(df['trd_dt'].map(str))
    df=df.set_index(['stkcd','trd_dt'])
    df=df.sort_index()
    df.to_csv(os.path.join(DCSV,tbname+'.csv'))
    df.to_pickle(os.path.join(DPKL,tbname+'.pkl'))

#TODO: read directly from sql rather than using read_raw

