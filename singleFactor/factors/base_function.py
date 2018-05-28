# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-27  22:07
# NAME:FT-base_function.py
from config import DCSV, DPKL
import os
import pandas as pd

# refer to the paper for operator
from singleFactor.factors.test_single_factor import test_single_factor
from tools import handle_duplicates

base_variables=[]

def ttm_adjust(df,col,q=4):
    df[col]=df[col].groupby('stkcd').apply(
        lambda s:s.rolling(q,min_periods=q).sum())
    return df

def adjust_result_format(df):
    '''
    虽然用了stkcd和report_period 作为主键，但是不同的report_period 对应的trd_dt
    可能相同，比如，asharefinancialindicator 中的000002.SZ，其2006-12-31 和
    2007-12-31的trd_dt 都是 2008-03-21
    '''
    result=df.reset_index().sort_values(['stkcd','trd_dt','report_period'])
    result=result.set_index(['trd_dt','stkcd'])[['result']].dropna()
    retult=handle_duplicates(result) # delete duplidates
    return result

def level(df,x,ttm=True):
    '''
    计算某个level的ttm
    df 是按季度的数据
    '''
    if ttm:
        df=ttm_adjust(df,x)
    df['result']=df[x]
    return adjust_result_format(df)

def x_pct_chg(df,x,ttm=True):
    '''
        d(x)/x
    percentage change in each accounting variable

    Args:
        df:按季度的数据
        x:
        ttm: True or False

    Returns:

    '''
    if ttm:
        df=ttm_adjust(df,x)
    df['result']=df[x].groupby('stkcd').apply(
        lambda s:s.pct_change())
    return adjust_result_format(df)

# base algorithm
def ratio_x_y(df,x,y,ttm=True):
    '''
    x/y
    financial ratio in x/y
    '''
    if ttm:
        df=ttm_adjust(df,x)
        df=ttm_adjust(df,y)
    df['result']=df[x]/df[y]
    return adjust_result_format(df)

def ratio_yoy_chg(df, x, y,ttm=True):
    '''
    d(x/y)
    year-to-year change in financial ratio
    '''
    if ttm:
        df=ttm_adjust(df,x)
        df=ttm_adjust(df,y)
    df['ratio']=df[x]/df[y]
    df['result']=df['ratio'].groupby('stkcd').apply(
        lambda s:s-s.shift(4))
    return adjust_result_format(df)

def ratio_yoy_pct_chg(df, x, y,ttm=True):
    '''
    d(x/y)/(x/y)
    year-to-year "percent" change in financial ratio
    '''
    if ttm:
        df = ttm_adjust(df, x)
        df = ttm_adjust(df, y)
    df['ratio']=df[x]/df[y]
    df['result']=df['ratio'].groupby('stkcd').apply(
        lambda s:s.pct_change(periods=4))
    return adjust_result_format(df)

def pct_chg_dif(df,x,y,ttm=True):
    '''
    d(x)/x -d(y)/y
    the difference between the percentage change in each accounting variable and
    the percentage change in a base variable
    '''
    if ttm:
        df=ttm_adjust(df,x)
        df=ttm_adjust(df,y)
    df['pct_chg_x']=df[x].groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['pct_chg_y']=df[y].groupby('stkcd').apply(
        lambda s:s.pct_change())
    df['result']=df['pct_chg_x']-df['pct_chg_y']
    return adjust_result_format(df)

def ratio_x_chg_over_lag_y(df,x,y,ttm=True):
    '''
    d(x)/lag(y)
    the change in each accounting variable scaled by a lagged base variable
    '''
    if ttm:
        df=ttm_adjust(df,x)
        df=ttm_adjust(df,y)
    df['x_chg']=df[x].groupby('stkcd').apply(lambda s:s-s.shift(1))
    df['lag_y']=df[y].groupby('stkcd').shift(1)
    df['result']=df['x_chg']/df['lag_y']
    return adjust_result_format(df)

#-------------------------------------------------------------------------------


if __name__ == '__main__':
    tbname = 'asharefinancialindicator'
    df = pd.read_pickle(os.path.join(DPKL, tbname + '.pkl'))
    x = 's_fa_ebit'
    y = 's_fa_tangibleasset'

    r1 = ratio_x_y(df, x, y)
    r2 = ratio_yoy_chg(df, x, y)
    r3 = ratio_yoy_pct_chg(df, x, y)
    r4 = x_pct_chg(df, x)
    r5 = pct_chg_dif(df, x, y)
    r6 = ratio_x_chg_over_lag_y(df, x, y)


def raw(df, col):
    result=df[[col]]
    result['trd_dt'] = df['trd_dt']
    result = result.reset_index().set_index(['trd_dt', 'stkcd'])[[col]].dropna()
    return result


def g_q(df,col):
    pass

def ttm(df,col,q=4):
    '''
    df 是按季度的财务指标
    Args:
        df:
        col:
        q: 季度数

    Returns:

    '''
    result=df[[col]].groupby('stkcd').apply(
        lambda x:x.rolling(period=q,min_periods=1).mean()
    )
    result['trd_dt']=df['trd_dt']
    result=result.reset_index().set_index(['trd_dt','stkcd'])[[col]].dropna()
    return result


def g_yoy(df, col, q=4):
    # growth ratio of year over year (4 quarters)
    result=df[[col]].groupby('stkcd').apply(
        lambda x:x.pct_change(periods=q,limit=q)
    )
    result['trd_dt']=df['trd_dt']
    result=result.reset_index().set_index(['trd_dt','stkcd'])[[col]].dropna()
    return result

def avg(df,col,q=4):
    # mean
    result=df[[col]].groupby('stkcd').apply(
        lambda x:x.rolling(q,min_periods=q).mean())
    result['trd_dt']=df['trd_dt']
    result=result.reset_index().set_index(['trd_dt','stkcd'])[[col]].dropna()
    return result


def test_base_function():
    tbname = 'asharefinancialindicator'
    df=pd.read_pickle(os.path.join(DPKL,tbname+'.pkl'))

    col='s_fa_ebitps'

    r_avg=avg(df,col)
    r_yoy=g_yoy(df, col)


def get_roe_growth_rate():
    # ROE 增长率
    tbname = 'asharefinancialindicator'
    df=pd.read_pickle(os.path.join(DPKL,tbname+'.pkl'))
    col='s_fa_roe'

    result=g_yoy(df,col)
    test_single_factor(result,result.columns[0])


def get_artrate():
    # 应收账款周转率
    tbname = 'asharefinancialindicator'
    df=pd.read_pickle(os.path.join(DPKL,tbname+'.pkl'))
    col='s_fa_arturn'
    result=raw(df,col)
    test_single_factor(result,result.columns[0])




