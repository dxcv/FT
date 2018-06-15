# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-15  13:54
# NAME:FT-derivatives.py
import pandas as pd

import os
from config import D_FT_DRV
from data.dataApi import read_local
from data.derivative.combine import get_dataspace



# ttm
balance=read_local('equity_selected_cashflow_sheet')

col='depr_fa_coga_dpba'

def _ttm(s):
    df=s.reset_index()
    del df['stkcd']
    df=df.set_index('report_period')
    df.columns=['raw']
    df['year_before']=df['raw'].shift(4)
    v=df.loc[df.index-pd.offsets.YearEnd(1)]['raw'].values
    df['last_year_end']=v
    df['ttm']=df['raw']-df['year_before']+df['last_year_end']
    return df['ttm']

def cal_ttm():
    ttm=balance[col].groupby('stkcd').apply(_ttm)











def cal_ebit():
    #息税前利润=净利润+所得税费用+财务费用   国泰安数据
    balance = read_local('equity_selected_income_sheet_q')
    ebit=balance['net_profit_incl_min_int_inc']+balance['inc_tax']+balance['fin_exp']

    balance=

    ebit.to_pickle(os.path.join(D_FT_DRV,'ebit.pkl'))

def cal_ebitda():
    #EBITDA=息税前利润+当期计提折旧与摊销  wind code generator EBITDA(反推法)
    fields=['net_profit_incl_min_int_inc','inc_tax','fin_exp',
            'depr_fa_coga_dpba','amort_intang_assets','amort_lt_deferred_exp']
    df=get_dataspace(fields)


    df['ebitda']=df['net_profit_incl_min_int_inc']+df['inc_tax']+df['fin_exp']+ \
        df['depr_fa_coga_dpba']+df['amort_intang_assets']+df['amort_lt_deferred_exp']
    return


cal_ebit()
