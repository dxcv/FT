# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-22  15:12
# NAME:xiangqi-sp.py
from config import DCC
from data.dataApi import get_indicator
from data.get_base import read_base
from tools import get_inter_index, convert_freq
import pandas as pd
import os

limit=300


def get_year_1():
    df=read_base('listdate')
    def _add_year_1(df):
        df['year_1']=df.index.get_level_values('trd_dt')<=df['listdate'][0]+pd.DateOffset(years=1)
        return df
    df=df.groupby('stkcd').apply(_add_year_1)
    df[['year_1']].to_pickle(os.path.join(DCC,'year_1.pkl'))


# get ret_m
df=read_base(['adjopen','adjclose'])


def _cal_ret_m_one_month(df):
    open=df.resample('M',level='trd_dt').first()['adjopen']
    close=df.resample('M',level='trd_dt').last()['adjclose']
    ret_m=(close-open)/open
    return ret_m[0]

def _cal_ret_m(df):
    result=df.groupby(pd.Grouper(freq='M',level='trd_dt')).apply(_cal_ret_m_one_month)
    return result

ret_m=df.groupby('stkcd').apply(_cal_ret_m)

print('a')



def func(s):
    s.groupby(pd.Grouper(freq='M',level='trd_dt')).apply()

    m=s.resample('M',level='trd_dt').last()

    result=(s+1).cumprod()
    return result


# ret_m=pctchange['stkcd'].groupby('stkcd').apply(lambda s:(s+1).cumprod())
# ret_m=pctchange.groupby('stkcd').apply(func)

# ret_m1=ret_m.groupby('stkcd').shift()


def get_sp():
    # oper_rev=get_indicator('equity_selected_income_sheet_q','oper_rev')
    df=read_base(['cap','oper_rev','type_st','year_1','wind_indcd', 'cap'])
    df['sp']=df['oper_rev']/df['cap']
    df=df.groupby('stkcd').ffill(limit=limit).dropna()
    df1=df.groupby('stkcd').resample('M',level='trd_dt').last()
    df2=df1[(df1.year_1==0) & (df1.type_st==0)]
    return df2




def factor_merge(df1, df2, keys=['stkcd', 'trd_dt']):
    '''
    df1, df2: DataFrame
        因子购建需要的数据
    去除含有ST标记或上市不满一年的股票

    '''
    data = pd.merge(df1, df2.reset_index(), on=keys, how='left')
    data = data.groupby(
        'stkcd').ffill().dropna()  # TODO: wrong!! there should be a thresh
    # TODO: warning,just use the last day to determine whether a stock is st or not is not proper
    data = data.groupby('stkcd').resample('M', on='trd_dt').last()
    #    data = data.set_index(['stkcd', 'trd_dt'], drop=False)
    data = data[(data.type_st == 0) & (data.year_1 == 0)]
    return data



# data=fdmt.join(sp,how='left')
#
# data=data.groupby('stkcd').ffill(limit=365).dropna()
# data=data.groupby('stkcd').resample('M',level='trd_dt').last()
#
#
# data = data.groupby('stkcd').resample('M', on='trd_dt').last()
#
# #TODO:use the same data to compare
#
#
#
# fd=get_indicator('equity_fundamental_info')
#
# member=fd['stkname']


