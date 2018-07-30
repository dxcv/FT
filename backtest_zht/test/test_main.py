# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-25  16:12
# NAME:FT_hp-test_main.py


import pandas as pd
from backtest_zht.main import run_backtest
from config import DIR_TMP
import os

signal=pd.read_pickle(r'E:\FT_Users\HTZhang\FT\singleFactor\signal\Q__roa.pkl')


run_backtest(signal,'roa',os.path.join(DIR_TMP,'roa'),start='2010',end='2015')


# from singleFactor.backtest_signal import save_result as rb

# rb(signal,'roa',os.path.join(DIR_TMP,'roa_cz'),start='2010',end='2015')

