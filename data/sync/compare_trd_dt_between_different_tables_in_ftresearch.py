# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-30  11:29
# NAME:FT-compare_trd_dt_between_different_tables_in_ftresearch.py
from data.dataApi import read_local, read_raw
import pandas as pd

def compare_():
    tb1='equity_selected_cashflow_sheet'
    tb2='equity_selected_cashflow_sheet_q'

    # tb2='equity_selected_balance_sheet'

    df1=read_local(tb1)['trd_dt']
    df2=read_local(tb2)['trd_dt']

    df=pd.concat([df1,df2],axis=1,keys=['date1','date2'])

    df=df.dropna()

    df['marker']=(df['date1']==df['date2'])

    print(df.shape)
    print(df['marker'].sum())


def _float2DateStr(x):
    if isinstance(x,(int,float)):
        x=str(x)

    if isinstance(x,str) and '.' in x:
        x=x.split('.')[0]
    return x


tbnames = [
    # 'equity_selected_balance_sheet',
    'equity_selected_cashflow_sheet',
    'equity_selected_cashflow_sheet_q',
    # 'equity_selected_income_sheet',
    # 'equity_selected_income_sheet_q',
]

ss=[]
for tbname in tbnames:
    df=read_raw(tbname)
    date_cols = ['trd_dt', 'ann_dt', 'report_period']
    for dc in date_cols:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc].map(_float2DateStr))

        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc].map(_float2DateStr))
    df=df.sort_values(['stkcd','report_period','trd_dt'])
    ss.append(df.set_index(['stkcd','report_period'])['ann_dt'])
    # dups=df.duplicated(subset=['stkcd','report_period'],keep='last')
    # print(tbname,df.shape[0],dups.sum())

dates_df=pd.concat(ss,axis=1,keys=tbnames)
dates_df['equal']=dates_df[tbnames[0]]==dates_df[tbnames[1]]
# dates_df=dates_df.dropna()

# dates_df['marker']=dates_df.apply(lambda s:(s==s[0]).sum(),axis=1)

# print(dates_df.shape[0],(dates_df['marker']!=5).sum())

# dates_df[dates_df['marker']!=5].sort_index().to_csv(r'e:\a\dates_df.csv')


#TODO: calculate q sheet


