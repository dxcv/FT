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


v = [-1, 5, 0, 0, 10, 0, -7]
v1 = [1, 0, 0, 0, 0, 0, 0]
v2 = [0, 1, 0, 0, 1, 0, 0]
v3 = [1, 1, 0, 0, 0, 0, 1]

s = pd.Series(v)
df = pd.DataFrame([v1, v2, v3], columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])


df.corrwith(df['a'])

df.corrwith(s)
df.corrwith(s[:3])
