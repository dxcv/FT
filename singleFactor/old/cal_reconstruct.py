# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-07  10:12
# NAME:FT-cal_reconstruct.py
from functools import reduce

import pandas as pd
import numpy as np

import xiangqi.data_clean as dc
import xiangqi.factor_test as ft
import os

from data.dataApi import read_local_pkl
from singleFactor.old.cal_tools import read_fields_map
from singleFactor.old import check_factor

TEST_DIR=r'E:\a\testdir'

#-==============================================================================
def factor_merge(df1, df2):
    '''
    df1, df2: DataFrame
        因子购建需要的数据
    去除含有ST标记或上市不满一年的股票
    '''
    keys = ['stkcd', 'trd_dt']
    data = pd.merge(df1, df2, on=keys, how='left')
    #TODO:data=pd.merge(df1,df2,on=keys,right_index=True,how='left')

    # data=data[-int(data.shape[0]/20):]#TODO: use a small sample to test the codes
    data = data.groupby('stkcd').ffill().dropna() #TODO: wrong!! there should be a thresh
    #TODO: warning,just use the last day to determine whether a stock is st or not is not proper
    data = data.groupby('stkcd').resample('M', on='trd_dt').last()
#    data = data.set_index(['stkcd', 'trd_dt'], drop=False)
    data = data[(data.type_st == 0) & (data.year_1 == 0)]
    return data

def _check(df, name):
    '''
    check single factor
    Args:
        df: pd.DataFrame,with only one column

    Returns:

    '''
    fdmt=pd.read_pickle(r'D:\zht\database\quantDb\internship\FT\fdmt.pkl')
    retn_1m=pd.read_pickle(r'D:\zht\database\quantDb\internship\FT\retn_1m.pkl')
    retn_1m_zz500=pd.read_pickle(r'D:\zht\database\quantDb\internship\FT\retn_1m_zz500.pkl')

    data=factor_merge(fdmt,df)
    # data=dm.factor_merge(fdmt,df)
    data=data.loc[:,['stkcd','trd_dt','wind_indcd','cap',name]]
    data['{}_raw'.format(name)]=data[name]
    # s_raw=data['oper_profit_raw'].describe()
    data=dc.clean(data,name)

    data=data.set_index(['trd_dt','stkcd'])
    data.index.names=['trade_date','stock_ID']
    signal_input=data[['{}_neu'.format(name)]]
    test_data=ft.data_join(retn_1m,signal_input)

    btic_des,figs1,btic_m=ft.btic(test_data,name)
    layer_des,figs2,layer_retn=ft.layer_result(test_data,retn_1m_zz500,name)

    path = os.path.join(TEST_DIR, name)
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path, name + '.csv'))

    btic_m.to_csv(os.path.join(path,'btic_m.csv'))
    btic_des.to_csv(os.path.join(path,'btic_des.csv'))
    layer_des.to_csv(os.path.join(path,'layer_des.csv'))
    layer_retn.to_csv(os.path.join(path,'layer_retn.csv'))

    for i,fig in enumerate(figs1+figs2):
        fig.savefig(os.path.join(path,'fig{}.png'.format(i)))

def get_local_df(dfname):
    df=read_local_pkl(dfname).reset_index().sort_values(['stkcd','trd_dt','report_period'])
    df=df[~df.duplicated(['stkcd','trd_dt'],keep='last')]
    return df


def get_dataspace(fields):
    fields_map=read_fields_map()
    if isinstance(fields,str): #only one field
        fields=[fields]

    dfnames=list(set([fields_map[f] for f in fields]))
    if len(dfnames)==1:
        df=read_local_pkl(dfnames[0])
    else:
        dfs=[get_local_df(dfname) for dfname in dfnames]
        df=reduce(lambda left,right:pd.merge(left,right,how='outer',on=['stkcd','trd_dt']),dfs)
        df=df.sort_values(['stkcd','trd_dt'])
        df=df.groupby('stkcd').ffill(limit=4) #TODO: how to set thresh? Test the difference,
    return df
#==========================================base function========================
def ttm_adjust(s):
    return s.groupby('stkcd').apply(
        lambda x:x.rolling(4,min_periods=4).sum())


def x_pct_chg(df, col, q=1, ttm=True,delete_negative=True):
    '''
        d(x)/x
    percentage change in each accounting variable

    Args:
        df:按季度的数据
        col:
        q:int,季度数，如果是求同比增长，则取q=4
        ttm: True or False

    Returns: df[['result']]

    '''
    df=df.copy()
    if delete_negative:
        df[col]=df[col].where(df[col]>0,np.nan)
        # df[col][df[col]<=0]=np.nan
    if ttm:
        df[col]=ttm_adjust(df[col])
    df['target']=df[col].groupby('stkcd').apply(lambda s:s.pct_change(periods=q))
    return df


#========================================cal_tools.py==============================
def check_g_yoy(df, col, name,q=4):
    '''
    yoy 增长率
    Args:
        df:
        col: 要检验的指标
        name: 保存的文件夹名
        q:int,q=4 表示yoy
    '''
    r_ttm=x_pct_chg(df,col,q=q,ttm=True)
    r=x_pct_chg(df,col,q=q,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,name)


#==========================================calculate_excel.py===================
def get_v_dp():
    name='V_dp'
    df=get_dataspace(['cash_div', 'tot_assets'])
    df[name]=df['cash_div']/df['tot_assets']
    _check(df[['stkcd','trd_dt',name]],name)


def get_dividendCover(): #TODO: dividend
    #股息保障倍数＝归属于母公司的净利润/最近 1 年的累计派现金额
    name='Q_dividendCover'
    col1='net_profit_excl_min_int_inc'
    col2='cash_div'
    df=get_dataspace([col1, col2])
    df[name]=df[col1]/df[col2]
    _check(df[['stkcd','trd_dt',name]],name)

def get_roe_growth_rate():#TODO: add indicators
    # ROE 增长率
    name='G_roe'
    col='s_fa_roe'
    df=get_dataspace(col)
    check_g_yoy(df,col,name)

get_roe_growth_rate()



