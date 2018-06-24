# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-23  20:53
# NAME:FT-a.py
import os
import pandas as pd

from config import FORWARD_TRADING_DAY
from data.dataApi import read_local



trading_m=read_local('trading_m')

window=2

close=pd.pivot_table(trading_m,values='close',index='trd_dt',columns='stkcd')
mom=close.pct_change(window)

b=trading_m['close'].groupby('stkcd').pct_change(periods=window)
type(b)
b.name
