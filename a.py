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


data=pd.read_pickle(r'e:\a\data.pkl')

monthly1=data.groupby('stkcd').resample('M',on='trd_dt').last().dropna()
