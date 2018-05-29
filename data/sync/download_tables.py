# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  13:25
# NAME:FT-download_from_ftresearch.py

import pymysql
import pandas as pd
import os
from config import DRAW, DCSV, DPKL
from data.dataApi import read_raw


def download_from_server(tbname,database='filesync'):
    try:
        db = pymysql.connect('192.168.1.140', 'ftresearch', 'FTResearch',
                             database,charset='utf8')
    except:
        db=pymysql.connect('localhost','root','root',database,charset='utf8')
    cur = db.cursor()
    query = 'SELECT * FROM {}'.format(tbname)
    cur.execute(query)
    table = cur.fetchall()
    cur.close()
    table = pd.DataFrame(list(table),columns=[c[0] for c in cur.description])
    table.to_csv(os.path.join(DRAW, '{}.csv'.format(tbname)))


#--------------------------- download ftresearch------------------------------------------
def download_ftresearch():
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

    for tbname in tbnames:
        download_from_server(tbname,database='ftresearch')
        print(tbname)

#-------------------------------download filesync-------------------------------
def download_filesync():
    tbnames=['ashareindicator','asharecalendar']
    for tbname in tbnames:
        download_from_server(tbname,'filesync')

#------------------------------adjust ftresearch-------------------------------

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

    def toDateStr(x):
        if isinstance(x,(int,float)):
            x=str(x)

        if '.' in x:
            x=x.split('.')[0]
        return x

    for tbname in tbnames:
        print(tbname)
        df=read_raw(tbname).reset_index(drop=True)
        date_cols=['trd_dt','ann_dt','report_period']
        for dc in date_cols:
            if dc in df.columns:
                df[dc]=pd.to_datetime(df[dc].map(toDateStr))

        df=df.sort_values(['stkcd','report_period','trd_dt'])
        df=df[~df.duplicated(subset=['stkcd','report_period'],keep='first')]
        df=df.groupby('stkcd').apply(
            lambda x:x.set_index('report_period').resample('Q').asfreq()
        )

        df=df.drop(['stkcd','create_time','update_time'],axis=1)

        df.to_csv(os.path.join(DCSV,tbname+'.csv'))
        df.to_pickle(os.path.join(DPKL,tbname+'.pkl'))

#---------------------------------adjust the format of sheet in filesync--------
def adjust_filesync(tbname):
    dateFields = ['report_period', 'trd_dt', 'ann_dt', 'holder_enddate',
                  'listdate','actual_ann_dt']
    # actual_ann_dt 指的是“更正公告的日期”
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
        # qrange=pd.date_range(x.index.min(), x.index.max(), freq='Q')
        # result2=x.reindex(pd.Index(qrange,name='report_period'))
        x=x.resample('Q').asfreq()
        return x

    #handle duplicates,keep the last one
    #keep the first one,since in real life,we can only trade based on the first
    #one.
    df=df.sort_values(['wind_code','report_period','trd_dt'])
    df=df[~df.duplicated(['wind_code','report_period'],keep='first')]
    df=df.groupby('wind_code').apply(_reindex_with_qrange)

    df=df.drop(['object_id','s_info_windcode','wind_code'],axis=1)
    df=df.reset_index().rename(columns={'wind_code':'stkcd'})

    df=df.set_index(['stkcd','report_period'])
    df=df.sort_index()

    #reorder the columns
    newcols=['trd_dt']+[c for c in df.columns if c!= 'trd_dt']
    df=df[newcols]

    df.to_csv(os.path.join(DCSV,tbname+'.csv'))
    df.to_pickle(os.path.join(DPKL,tbname+'.pkl'))

    #TODO:notice that do not apply sort and dropna on df
    #TODO:notice that there are some NaT in index

def convert_asharefinancialindicator():
    tbname='asharefinancialindicator'
    adjust_filesync(tbname)


#TODO: there is some invalid codes at the end of the index
#TODO:just use these tables to calculate new indicators



