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


directory=r'D:\zht\database\quantDb\internship\FT\singleFactor\indicators\financial'

fns=os.listdir(directory)

fns=[fn for fn in fns if fn.endswith('.pkl')]


for fn in fns:
    df=pd.read_pickle(os.path.join(directory,fn))
    print(df.columns[0])

