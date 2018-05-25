# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-21  17:07
# NAME:FT-dataApi.py
import datetime
import os
import time

import sqlalchemy

import pymysql
import pandas as pd
import numpy as np
from datetime import timedelta

from config import START, END, DRAW
from data.database_api import database_api as dbi

from data.prepare import pre_process


def read_from_sql(tbname, cols=None):
    db = pymysql.connect('192.168.1.140', 'ftresearch', 'FTResearch',
                         'ftresearch')
    cur = db.cursor()
    if cols:
        fields = ','.join(cols)
        query = 'SELECT ' + fields + ' FROM ' + tbname
    else:
        query='SELECT * FROM {}'.format(tbname)
    cur.execute(query)
    table = cur.fetchall()
    table = pd.DataFrame(list(table))
    table.columns = cols
    return table

def _get_indicator(tbname,indname,freq='M'):
    cols=['stkcd','trd_dt',indname]
    table=read_from_sql(tbname, cols)
    table['trd_dt']=pd.to_datetime(table['trd_dt'])
    table=pd.pivot_table(table,values=indname,columns='stkcd')
    return table

def _read_local_db(tbname,field=None,start=START,end=END):
    start=datetime.datetime.strptime(start,'%Y-%m-%d')
    end=datetime.datetime.strptime(end,'%Y-%m-%d')
    df=pd.read_pickle(os.path.join(DRAW, tbname + '.pkl'))
    df=df[(df['trd_dt']>=start) & (df['trd_dt']<=end)]
    df=df.set_index(['trd_dt','stkcd'])
    if field:
        if isinstance(field,str):
            df=df[[field]]
        else:
            df=df[field]
    return df

def _read_from_sql(tbname,field=None,start=START,end=END):
    if field:# only get the given fields
        if isinstance(field, str):
            data=dbi.get_stocks_data(tbname, [field], start, end)[field]
        else:
            data=dbi.get_stocks_data(tbname, field, start, end)
        return data
    else:# read the whole table
        raise NotImplementedError

def get_indicator(tbname, field=None, start=START, end=END, prep=True,db='local'):
    if db=='local':#read local csv
        data=_read_local_db(tbname,field,start,end)
    else:
        #read from sql database
        data=_read_from_sql(tbname,field,start,end)
    if prep:
        # TODO: for rolling indicators,we shouldn't delete the
        data=pre_process(data)
    return data


def read_local(tbname, indicators=None):
    dateFields = ['report_period', 'trd_dt', 'ann_dt', 'holder_enddate',
                  'listdate']

    df = pd.read_csv(os.path.join(DRAW, tbname + '.csv'), index_col=0)
    for datef in dateFields:
        if datef in df.columns:
            df[datef] = pd.to_datetime(df[datef])

    # qstart = df['report_period'].min()
    # qend = df['report_period'].max()
    # qrange = pd.date_range(qstart, qend, freq='Q')
    # df1=df.groupby('stkcd').apply(lambda df:df.set_index('report_period').reindex(qrange))
    # df1.groupby('stkcd').apply(lambda x:print(x.shape[0]))
    df = df.set_index(['stkcd', 'trd_dt'])
    df = df.sort_values('report_period')


    if indicators is None:
        return df
    else:
        if isinstance(indicators, str):
            return df[[indicators, 'report_period']]
        else:
            return df[indicators + ['report_period']]





# if __name__ == '__main__':
#     oper_rev = get_indicator('equity_selected_income_sheet_q', 'oper_rev')
#     df = get_indicator('equity_selected_income_sheet_q',
#                        ['oper_rev', 'selling_dist_exp'])