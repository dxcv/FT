# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-23  20:53
# NAME:FT-a.py
import os
import pandas as pd

from config import FORWARD_TRADING_DAY
from data.dataApi import read_local, read_from_sql

df=pd.read_csv(r'E:\FT_Users\HTZhang\FT\singleFactor\combine\combine\mixed_mixed\hedged_returns.csv',index_col=0,parse_dates=True)

mom=pd.read_pickle(r'E:\FT_Users\HTZhang\FT\singleFactor\indicators\T__mom_20.pkl')

sub=mom['2015-08']

trading = read_local('equity_selected_trading_data')

adjclose = pd.pivot_table(trading, values='adjclose', index='trd_dt',
                          columns='stkcd')

subclose=adjclose['2015-07']

