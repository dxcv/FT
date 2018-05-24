# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-22  09:25
# NAME:FT-check_example.py

import sys
# sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib') # 自定义的函数
import pandas as pd
from data.database_api import database_api as dbi
import xiangqi.data_merge as dm
import xiangqi.data_clean as dc
import xiangqi.factor_test as ft
import os

oper_rev = dbi.get_stocks_data('equity_selected_income_sheet_q', ['oper_rev'],
                               '2004-01-01', '2018-03-01')



oper_rev['oper_rev_ttm'] = oper_rev.groupby('stkcd')[['oper_rev']].apply(
        lambda x: x.rolling(4, min_periods=4).sum())

store=pd.HDFStore(r'\\Ft-research\e\Share\Alpha\FYang\factors\test_data.h5')

fdmt = store['fundamental_info']
retn_1m=store['retn_1m']
retn_1m_zz500=store['retn_1m_zz500']
store.close()

data = dm.factor_merge(fdmt, oper_rev) # 数据合并，并去除含有ST标记上市不满一年的股票
data = data.loc[:, ['stkcd', 'trd_dt', 'wind_indcd', 'cap', 'oper_rev_ttm']]
data['SP_TTM_raw'] = data['oper_rev_ttm'] / (10000 * data['cap'])


s_raw = data['SP_TTM_raw'].groupby(level=1).describe()
data.drop(pd.to_datetime(['2005-01-31', '2005-02-28', '2005-03-31']), level=1, inplace=True) # 删除样本不全的数据
data = dc.clean(data, 'SP_TTM') # 去极值、标准化、行业与市值中性化
data = data.set_index(['trd_dt', 'stkcd'])
data.index.names = ['trade_date', 'stock_ID'] # 使index的名字与公司框架要求的一致，从而方便后期数据的聚合

signal_input = data[['SP_TTM_neu']]
test_data = ft.data_join(retn_1m, signal_input)

btic_des = ft.btic(test_data, 'SP_TTM')
layer_des = ft.layer_result(test_data, retn_1m_zz500, 'SP_TTM')

