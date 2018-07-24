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



for Y in [str(i) for i in range(2010,2019)]:
    print(Y,(1+df[Y]).cumprod().values[-1])
