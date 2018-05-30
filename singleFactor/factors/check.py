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

def check_factor(df,name):
    '''

    Args:
        df:pd.DataFrame,with only one column
        col:
        name:

    Returns:

    '''
    drct = r'D:\zht\database\quantDb\internship\FT\singleFactor\result'
    path = os.path.join(drct, name)
    if not os.path.exists(path):
        os.makedirs(path)

    store=pd.HDFStore(r'\\Ft-research\e\Share\Alpha\FYang\factors\test_data.h5')
    fdmt = store['fundamental_info']
    retn_1m=store['retn_1m']
    retn_1m_zz500=store['retn_1m_zz500']
    store.close()

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

    btic_des,figs1=ft.btic(test_data,col)
    layer_des,figs2=ft.layer_result(test_data,retn_1m_zz500,col)

    btic_des.to_csv(os.path.join(path,'btic_des.csv'))
    layer_des.to_csv(os.path.join(path,'layer_des.csv'))

    for i,fig in enumerate(figs1+figs2):
        fig.savefig(os.path.join(path,'fig{}.png'.format(i)))

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