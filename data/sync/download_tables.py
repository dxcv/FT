# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  13:25
# NAME:FT-download_from_ftresearch.py
import time

import pymysql
import pandas as pd
import os
import numpy as np


from config import DRAW, DCSV, DPKL

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

def download_from_server(tbname,database='filesync'):
    try:
        db = pymysql.connect('192.168.1.140', 'ftresearch', 'FTResearch',
                             database)
    except:
        db=pymysql.connect('')
    cur = db.cursor()
    query = 'SELECT * FROM {}'.format(tbname)
    cur.execute(query)
    table = cur.fetchall()
    cur.close()
    table = pd.DataFrame(list(table),columns=[c[0] for c in cur.description])
    table.to_csv(os.path.join(DRAW, '{}.csv'.format(tbname)))

def read_raw(tbname):
    return pd.read_csv(os.path.join(DRAW, tbname + '.csv'))


def raw2csvpkl(tbname):
    dateFields = ['report_period', 'trd_dt', 'ann_dt', 'holder_enddate',
                  'listdate']
    df = pd.read_csv(os.path.join(DRAW, tbname + '.csv'), index_col=0)

    #adjust column format
    df.columns=[str.lower(c) for c in df.columns]

    #adjust date format
    for datef in dateFields:
        if datef in df.columns:
            df[datef]=df[datef].map(lambda x:str(x)[:8])
            df[datef] = pd.to_datetime(df[datef])

    #map ann_date to trading date
    calendar=read_raw('asharecalendar')
    trd_dt=pd.to_datetime(calendar['TRADE_DAYS'].map(str)).drop_duplicates().sort_values()
    df['trd_dt']=df['ann_dt'].map(lambda x:trd_dt.values[trd_dt.searchsorted(x)[0]])

    # reindex with qrange
    def _reindex_with_qrange(x):
        x=x.set_index('report_period')
        qrange=pd.date_range(x.index.min(), x.index.max(), freq='Q')
        x=x.reindex(pd.Index(qrange,name='report_period'))
        return x

    #handle duplicates,keep the last one
    df=df[~df.duplicated(['wind_code','report_period'],keep='last')]
    df=df.sort_values(['wind_code','report_period'])
    df=df.groupby('wind_code').apply(_reindex_with_qrange)

    df=df.drop(['object_id','s_info_windcode','wind_code'],axis=1)
    df=df.reset_index().rename(columns={'wind_code':'stkcd'})

    # set ['trd_dt','stkcd'] as index
    df=df.set_index(['trd_dt','stkcd'])

    #reorder the columns
    newcols=['report_period']+[c for c in df.columns if c!= 'report_period']
    df=df[newcols]

    df.to_csv(os.path.join(DCSV,tbname+'.csv'))
    df.to_pickle(os.path.join(DPKL,tbname+'.pkl'))

    #TODO:notice that do not apply sort and dropna on df
    #TODO:notice that there are some NaT in index

def download_raw():
    tbnames=['ashareindicator','asharecalendar']
    for tbname in tbnames:
        download_from_server(tbname,'filesync')

def convert_raw():
    tbname='asharefinancialindicator'
    raw2csvpkl(tbname)


download_raw()
convert_raw()



