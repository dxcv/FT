# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-24  16:27
# NAME:FT-check.py

import pandas as pd
import xiangqi.data_merge as dm
import xiangqi.data_clean as dc
import xiangqi.factor_test as ft
import os

from config import SINGLE_D_RESULT

def prepare():
    store = pd.HDFStore(r'D:\zht\database\quantDb\internship\FT\test_data.h5')
    fdmt = store['fundamental_info']
    retn_1m = store['retn_1m']
    retn_1m_zz500 = store['retn_1m_zz500']
    store.close()

    fdmt.to_pickle(os.path.join(r'D:\zht\database\quantDb\internship\FT\fdmt.pkl'))
    retn_1m.to_pickle(os.path.join(r'D:\zht\database\quantDb\internship\FT\retn_1m.pkl'))
    retn_1m_zz500.to_pickle(os.path.join(r'D:\zht\database\quantDb\internship\FT\retn_1m_zz500.pkl'))

# prepare()

def _check(df,projName):
    '''
    check single factor
    Args:
        df: pd.DataFrame,with only one column

    Returns:

    '''
    fdmt=pd.read_pickle(r'D:\zht\database\quantDb\internship\FT\fdmt.pkl')
    retn_1m=pd.read_pickle(r'D:\zht\database\quantDb\internship\FT\retn_1m.pkl')
    retn_1m_zz500=pd.read_pickle(r'D:\zht\database\quantDb\internship\FT\retn_1m_zz500.pkl')

    col=df.columns[0]
    data=dm.factor_merge(fdmt,df)
    data=data.loc[:,['stkcd','trd_dt','wind_indcd','cap',col]]
    data['{}_raw'.format(col)]=data[col]
    # s_raw=data['oper_profit_raw'].describe()
    data=dc.clean(data,col)

    data=data.set_index(['trd_dt','stkcd'])
    data.index.names=['trade_date','stock_ID']
    signal_input=data[['{}_neu'.format(col)]]
    test_data=ft.data_join(retn_1m,signal_input)

    btic_des,figs1,btic_m=ft.btic(test_data,col)
    layer_des,figs2,layer_retn=ft.layer_result(test_data,retn_1m_zz500,col)

    path = os.path.join(SINGLE_D_RESULT, projName)
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path,projName+'.csv'))

    btic_m.to_csv(os.path.join(path,'btic_m.csv'))
    btic_des.to_csv(os.path.join(path,'btic_des.csv'))
    layer_des.to_csv(os.path.join(path,'layer_des.csv'))
    layer_retn.to_csv(os.path.join(path,'layer_retn.csv'))

    for i,fig in enumerate(figs1+figs2):
        fig.savefig(os.path.join(path,'fig{}.png'.format(i)))

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

def check_factor(df,projName):
    df=_change_index(df)
    _check(df,projName)


def check_factor1(df, name):
    drct = r'D:\zht\database\quantDb\internship\FT\singleFactor\result'
    path = os.path.join(drct, name)
    if not os.path.exists(path):
        os.makedirs(path)

    store=pd.HDFStore(r'\\Ft-research\e\Share\Alpha\FYang\factors\test_data.h5')
    fdmt = store['fundamental_info']
    retn_1m=store['retn_1m']
    retn_1m_zz500=store['retn_1m_zz500']
    store.close()

    data=dm.factor_merge(fdmt,df)
    data=data.loc[:,['stkcd','trd_dt','wind_indcd','cap',name]]
    data['{}_raw'.format(name)]=data[name]
    # s_raw=data['oper_profit_raw'].describe()
    data=dc.clean(data,name)

    data=data.set_index(['trd_dt','stkcd'])
    data.index.names=['trade_date','stock_ID']
    signal_input=data[['{}_neu'.format(name)]]
    test_data=ft.data_join(retn_1m,signal_input)

    btic_des,figs1=ft.btic(test_data,name)
    layer_des,figs2=ft.layer_result(test_data,retn_1m_zz500,name)

    btic_des.to_csv(os.path.join(path,'btic_des.csv'))
    layer_des.to_csv(os.path.join(path,'layer_des.csv'))

    for i,fig in enumerate(figs1+figs2):
        fig.savefig(os.path.join(path,'fig{}.png'.format(i)))
