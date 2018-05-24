# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-24  14:07
# NAME:FT-sp1.py
from data.get_base import read_base
import pandas as pd
from data.database_api import database_api as dbi
import xiangqi.data_merge as dm
import xiangqi.data_clean as dc
import xiangqi.factor_test as ft

df = read_base(['cap', 'oper_rev'])
df['sp'] = df['oper_rev'] / df['cap']
sp=df[['sp']]



store=pd.HDFStore(r'\\Ft-research\e\Share\Alpha\FYang\factors\test_data.h5')

fdmt = store['fundamental_info']
retn_1m=store['retn_1m']
retn_1m_zz500=store['retn_1m_zz500']
store.close()


sp=sp.reset_index()
data=dm.factor_merge(fdmt,sp)

data=data.loc[['stkcd','trd_dt','wind_indcd','cap','sp']]
data['sp_raw']=data['sp']




data=dc.clean(data,'sp')
data=data.set_index(['trd_dt','stkcd'])

signal_input=data[['sp']]
test_data=ft.data_join(retn_1m,signal_input)

bitic_des=ft.btic(test_data,'sp')
layer_des=ft.layer_result(test_data,retn_1m_zz500,'sp')
