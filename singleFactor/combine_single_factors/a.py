# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-03  15:52
# NAME:FT_hp-a.py


import pandas as pd
import numpy as np


import os

from backtest_zht.main_class import Backtest
from config import DIR_TMP

signal=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\tmp\signal.pkl')

name='aaa'
Backtest(signal,name,directory=os.path.join(DIR_TMP,name))
