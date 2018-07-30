# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-29  13:09
# NAME:FT_hp-test_main_class.py

import pandas as pd
from backtest_zht.main_class import Backtest
from backtest_zht.main import run_backtest
from config import DIR_TMP
import os


signal=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\tmp\signal_500_5_cumprod_ret_200.pkl')

for i in range(3):
    name='test{}'.format(i)
    Backtest(signal,name,os.path.join(DIR_TMP,name),start='2009')
    print(i)



