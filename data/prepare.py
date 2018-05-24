# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  10:58
# NAME:FT-prepare.py
from config import START, END
from tools import monitor
import pandas as pd



@monitor
def check_format(x):
    if not isinstance(x,pd.DataFrame):
        raise TypeError('"{}" is not allowed'.format(type(x)))
    return x

@monitor
def check_index_names(x):
    names=['trd_dt','stkcd']
    if x.index.names!=names:
        raise IndexError('"{}" is invalid index name'.format(x.index.names))
    return x

@monitor
def handle_duplicates(x):
    #keep the last duplicated value
    if x.index.has_duplicates:
        print('Series has duplicated index and the last one will be kept')
        x=x[~x.index.duplicated(keep='last')]
    return x

@monitor
def check_order(x):
    if not x.index.is_monotonic:
        x=x.sort_index()
        print('index is not monotonic,it has been converted to a sorted index')
    return x

def pre_process(x):
    x = check_format(x)
    x = check_index_names(x)
    x = handle_duplicates(x)
    x = check_order(x)
    return x