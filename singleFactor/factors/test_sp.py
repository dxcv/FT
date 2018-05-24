# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-24  16:15
# NAME:FT-test_sp.py


import pandas as pd
from data.database_api import database_api as dbi
import xiangqi.data_merge as dm
import xiangqi.data_clean as dc
import xiangqi.factor_test as ft
import os

from data.get_base import read_base

start='2004-01-01'
end='2018-03-01'
name='sp'

drct=r'D:\zht\database\quantDb\internship\FT\singleFactor\result'
path=os.path.join(drct,name)
if not os.path.exists(path):
    os.makedirs(path)


df = read_base(['cap', 'oper_rev'])
df['sp'] = df['oper_rev'] / df['cap']
sp=df[['sp']]


# oper_profit=dbi.get_stocks_data('equity_selected_income_sheet_q',['oper_profit'],
#                             start,end)

def test(df):
    store=pd.HDFStore(r'\\Ft-research\e\Share\Alpha\FYang\factors\test_data.h5')
    fdmt = store['fundamental_info']
    retn_1m=store['retn_1m']
    retn_1m_zz500=store['retn_1m_zz500']
    store.close()

    data=dm.factor_merge(fdmt,sp)



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
