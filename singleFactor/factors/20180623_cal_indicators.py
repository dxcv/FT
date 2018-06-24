# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-23  18:13
# NAME:FT-20180623_cal_indicators.py

'''
idea:
    1. build a mould by use daily trading data
    2. put all the data into the mould
      dfs to put in mould:
        1) three sheet
        2) trading data
        3) div_cash data
    3. ffill with a limit
    4. calculate new indicator
    5. filter with st and young_1year
    6. output


'''
import os
from functools import reduce
import pandas as pd
from config import D_FT_RAW
from data.dataApi import read_local
from tools import number2dateStr


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
        'equity_selected_cashflow_sheet_q',
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



def get_standard_index():
    trading=read_local('equity_selected_trading_data')
    return trading.index

tbnames=['equity_selected_balance_sheet',
         'equity_selected_cashflow_sheet_q',
         'equity_selected_income_sheet_q',
         'equity_selected_trading_data',
         'equity_cash_dividend'
         ]

standard_index=get_standard_index()
dfs=[]
for tb in tbnames:
    df=pd.read_pickle(os.path.join(r'D:\zht\database\quantDb\internship\FT\database\ftresearch_based\adjusted\pkl',
                                   tb))
    df=df.reset_index().set_index(['stkcd','trd_dt'])
    print(tb)
    # df=read_local(tb).reset_index().set_index(['stkcd','trd_dt'])
    print(df.index.has_duplicates)
data=pd.concat(dfs,axis=1)
data.info()


