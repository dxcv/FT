# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-06  17:53
# NAME:FT-a.py

import pandas as pd
from backtest.main import quick
import numpy as np
from config import DIR_BACKTEST_RESULT
from data.dataApi import read_local

import os


name = 'C__est_bookvalue_FT24M_to_close_g_20'

monthly_check=pd.read_pickle(r'E:\FT_Users\HTZhang\tmp\monthly_check.pkl')

directory = os.path.join(DIR_BACKTEST_RESULT, name)

tmp = pd.read_csv(os.path.join(directory, 'signal.csv'), index_col=0,
                     parse_dates=True)


signal=pd.pivot_table(monthly_check,values=name,index='month_end',columns='stkcd')
signal=signal.reindex(pd.date_range(start=signal.index[0],end=signal.index[-1]))
signal=signal.ffill(limit=30)
signal=signal.reindex(tmp.index)

results,fig=quick(signal,fig_title='test',start='2010')

fig.show()






