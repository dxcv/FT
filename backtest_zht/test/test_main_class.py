# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-29  13:09
# NAME:FT_hp-test_main_class.py

import pandas as pd
from backtest_zht.main_class import Backtest
from config import DIR_TMP
import os


signal=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\singleFactor\signal\G_pct_12__tot_oper_rev.pkl')

name='test'
directory=os.path.join(DIR_TMP,name)

start='2018-01'
Backtest(signal,name,directory,start=start)



