# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-24  14:55
# NAME:FT-calculate_factors.py
from data.get_base import read_base
import data.database_api.database_api as dbi
from singleFactor.factors.test_single_factor import test
import pandas as pd

#同比增长比率
from tools import handle_duplicates


def test_ratio_category(tbname,name):
    ratio_name='{}__pct'.format(name)
    df=dbi.get_stocks_data(tbname,[name])
    df[ratio_name]=df.groupby('stkcd').pct_change()
    df=df[[ratio_name]]
    test(df,ratio_name)

#单季度营业利润同比增长率
tbname='equity_selected_income_sheet_q'
name='oper_profit'
test_ratio_category(tbname,name)

#单季度净利润同比增长率
tbname='equity_selected_income_sheet_q'
name='net_profit_excl_min_int_inc'
test_ratio_category(tbname,name)

#单季度营业收入同比增长率
tbname='equity_selected_income_sheet_q'
name='oper_rev'
test_ratio_category(tbname,name)

#每股收益同比增长率
cap_stk=dbi.get_stocks_data('equity_selected_balance_sheet',['cap_stk'])
income=dbi.get_stocks_data('equity_selected_income_sheet',['net_profit_excl_min_int_inc'])
cap_stk=handle_duplicates(cap_stk)
income=handle_duplicates(income)
df=pd.concat([cap_stk,income],axis=1)
df['income_per_share']=df['cap_stk']/df['net_profit_incl_min_int_inc']
df['income_per_share_pct']=df['income_per_share'].pct_change()
test(df['income_per_share_pct'],'income_per_share_pct')

#经营现金流增长率
tbname='equity_selected_cashflow_sheet'
name='net_cash_flows_oper_act'
test_ratio_category(tbname,name)

#净利润过去 5 年历史增长率
income=dbi.get_stocks_data('equity_selected_income_sheet',['net_profit_excl_min_int_inc'])
income['income_rolling_5']=income[['net_profit_excel_min_int_inc']].groupby('stkcd').apply(
    lambda x:x.rolling(5,min_periods=5).sum())
income['income_rolling_5_test']=income.groupby('stkcd')[['net_profit_excel_min_int_inc']].apply(
    lambda x:x.rolling(5,min_periods=5).sum())

test()
