# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-28  15:48
# NAME:FT-Q.py
from data.dataApi import read_local
from data.ftresearch.adjust import read_cache
import pandas as pd


balance=read_local('equity_selected_balance_sheet')
cash=read_local('equity_selected_cashflow_sheet_q')
income=read_local('equity_selected_income_sheet_q')


comb=pd.concat([balance['trd_dt'],cash['trd_dt'],income['trd_dt']],axis=1,
               keys=['balance','cash','income'])

w=comb[~((comb['balance']==comb['cash']) & (comb['balance']==comb['income']))]

a=w.dropna(how='all')
