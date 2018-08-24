# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-06  17:53
# NAME:FT-a.py

import pandas as pd
import os


ret_new=pd.read_csv(r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_signal_backtest\500_iw2_cw3_3_criteria3_100_1\hedged_returns.csv',index_col=0,parse_dates=True)
ret_old=pd.read_csv(r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_signal_backtest_backup\500_iw2_cw3_3_criteria3_100_1\hedged_returns.csv',index_col=0,parse_dates=True)

comb=pd.concat([ret_new.iloc[:,0],ret_old.iloc[:,0]],axis=1,keys=['new','old'])

comb=comb.dropna()

(1+comb).cumprod().plot().get_figure().show()
