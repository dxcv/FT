# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-21  17:07
# NAME:FT-dataApi.py
import datetime
import os
import pandas as pd
from config import DRAW, DPKL


def read_raw(tbname):
    return pd.read_csv(os.path.join(DRAW, tbname + '.csv'),index_col=0)

def read_local(tbname, col=None):
    df=pd.read_pickle(os.path.join(DPKL,tbname+'.pkl'))
    if col:
        if isinstance(col, str):# read only one column
            return df[[col]]
        else: #read multiple columns
            return df[col]
    else: # read all columns
        return df


# div=read_local('equity_cash_dividend')
