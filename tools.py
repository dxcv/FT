# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  10:57
# NAME:FT-tools.py

import time
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def monitor(func):
    def wrapper(*args,**kwargs):
        print('{}   starting -> {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),
                                           func.__name__))
        return func(*args,**kwargs)
    return wrapper

def get_inter_index(s1, s2):
    interInd=s1.index.intersection(s2.index)
    s1=s1.reindex(interInd)
    s2=s2.reindex(interInd)
    return s1, s2

def _convert_freq(x,freq,thresh):
    x=x.groupby(pd.Grouper(freq=freq,level='trd_dt')).last()
    # TODO: ffill whould only be used on indicators from financial report.
    # TODO: pay attention to cash_div and counts in
    x=x.ffill(limit=thresh)
    return x

def convert_freq(x, freq='M', thresh=12):
    newdf=x.groupby('stkcd').apply(_convert_freq, freq, thresh)
    newdf=newdf.swaplevel().sort_index()
    return newdf

def handle_duplicates(df):
    return df[~df.index.duplicated(keep='first')]

def number2dateStr(x):
    if x:
        if isinstance(x,(int,float)):
            x=str(x)

        if '.' in x:
            x=x.split('.')[0]
        return x

def daily2monthly(df):
    '''

    Args:
        df:DataFrame,contains column ['stkcd','trd_dt']

    Returns:

    '''
    df=df.sort_values(['stkcd','trd_dt'])
    monthly=df[(df['stkcd']==df['stkcd'].shift(-1)) &
                     (df['trd_dt'].dt.month!=df['trd_dt'].shift(-1).dt.month)]
    monthly=monthly.dropna(how='all')
    monthly['month_end']=monthly['trd_dt']+MonthEnd(0)
    return monthly
