# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  09:11
# NAME:FT-new_check_factors.py
import multiprocessing

from config import SINGLE_D_INDICATOR, SINGLE_D_RESULT
import os
import pandas as pd

import xiangqi.data_clean as dc
import xiangqi.factor_test as ft
from data.dataApi import read_local
from tools import monitor


def _change_index(df):
    '''
        虽然用了stkcd和report_period 作为主键，但是不同的report_period 对应的trd_dt
    可能相同，比如，asharefinancialindicator 中的000002.SZ，其2006-12-31 和
    2007-12-31的trd_dt 都是 2008-03-21
    Args:
        df:pd.DataFrame,with the index as ['stkcd','report_period'],and there
        should be a column named 'trd_dt'

    Returns:DataFrame,with only one column and the index is ['stkcd','trd_dt']

    '''
    if isinstance(df['trd_dt'],pd.DataFrame):
        '''如果df里边有多个名为trd_dt的列，取日期最大的那个'''
        trd_dt_df=df['trd_dt']
        df=df.drop('trd_dt',axis=1)
        df['trd_dt']=trd_dt_df.max(axis=1)

    df=df.reset_index().sort_values(['stkcd','trd_dt','report_period'])
    # 如果在相同的trd_dt有不同的report_period记录，取report_period较大的那条记录
    df=df[~df.duplicated(['stkcd','trd_dt'],keep='last')]
    df=df.set_index(['stkcd','trd_dt']).sort_index()[['target']].dropna()
    return df

def change_index(df):
    df = df.reset_index().sort_values(['stkcd', 'trd_dt', 'report_period'])
    # 如果在相同的trd_dt有不同的report_period记录，取report_period较大的那条记录
    df = df[~df.duplicated(['stkcd', 'trd_dt'], keep='last')]
    del df['report_period']
    df=df.set_index(['stkcd','trd_dt']).dropna()
    return df


def factor_merge(df1, df2):
    '''
    df1, df2: DataFrame
        因子购建需要的数据
    去除含有ST标记或上市不满一年的股票
    '''
    # data=df1.join(df2,how='left')
    data = pd.merge(df1.reset_index(), df2.reset_index(), on=['stkcd','trd_dt'], how='left')
    data=data[(~data['type_st']) & (~ data['young_1year'])] # 剔除st 和上市不满一年的数据
    #TODO： 涨跌停剔除，
    # data=data[-int(data.shape[0]/20):]#TODO: use a small sample to test the codes
    data=data.groupby('stkcd').ffill(limit=400).dropna() #TODO: 向前填充最最多400个交易日
    # 把日度变为月度数据，减小后边计算量
    data = data.groupby('stkcd').resample('M', on='trd_dt').last().dropna(how='all')
    data.index.names=['stkcd','month_end'] #resample 得到的是calendar date,mont_end,注意此处的index使用的是month_end 而不是trd_dt
    # data =data.reset_index(drop=True).set_index(['stkcd','trd_dt'])
    return data

#TODO:
#fixme
#Review

def _check(df):
    '''
    check single factor
    Args:
        df: pd.DataFrame,with only one column,and the index is ['stkcd','trd_dt']

    Returns:

    '''
    fdmt=read_local('equity_fundamental_info')
    ret_1m=read_local('trading_m')['ret_1m']
    zz500_ret_1m=read_local('indice_m')['zz500_ret_1m']

    col=df.columns[0]
    data=factor_merge(fdmt,df)
    
    data=data[['wind_indcd','cap',col]]
    data=dc.clean(data,col) # 去极值，标准化，中性化

    # data.index.names=['trade_date','stock_ID']
    signal_input=data[['{}_neu'.format(col)]]
    test_data=ft.data_join(ret_1m,signal_input)

    # delete those month with too small sample
    test_data=test_data.groupby('month_end').filter(lambda x:x.shape[0]>50)
    if test_data.shape[0]>0:
        btic_des,figs1,btic_m=ft.btic(test_data,col)
        layer_des,figs2,layer_retn=ft.layer_result(test_data,zz500_ret_1m,col)

        path = os.path.join(SINGLE_D_RESULT, col)
        if not os.path.exists(path):
            os.makedirs(path)

        df.to_csv(os.path.join(path, col + '.csv'))

        btic_m.to_csv(os.path.join(path,'btic_m.csv'))
        btic_des.to_csv(os.path.join(path,'btic_des.csv'))
        layer_des.to_csv(os.path.join(path,'layer_des.csv'))
        layer_retn.to_csv(os.path.join(path,'layer_retn.csv'))

        for i,fig in enumerate(figs1+figs2):
            fig.savefig(os.path.join(path,'fig{}.png'.format(i)))

#review:daf

#wrong dafadga



@monitor
def check_factors():
    fns=os.listdir(os.path.join(SINGLE_D_INDICATOR))
    fns=[fn for fn in fns if fn.endswith('.pkl')]
    for fn in fns:
        df=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR,fn))
        df=change_index(df)
        _check(df)

def task(fn):
    df=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR,fn))
    df=change_index(df)
    _check(df)


fns=os.listdir(SINGLE_D_INDICATOR)
task(fns[2])

# if __name__ == '__main__':
#     fns=os.listdir(SINGLE_D_INDICATOR)
#     fns=[fn for fn in fns if fn.endswith('.pkl')]
#     pool=multiprocessing.Pool(2)
#     pool.map(task,fns)
