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

df=pd.read_pickle(r'E:\FT_Users\HTZhang\FT\singleFactor\indicators\Q__downturnRisk.pkl')

df.index.name

