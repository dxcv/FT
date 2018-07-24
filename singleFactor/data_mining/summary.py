# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-18  08:32
# NAME:FT_hp-summary.py

import os
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from config import DIR_DM_BACKTEST, DIR_TMP
import time
import multiprocessing


def get_ret(name):
    return pd.read_csv(os.path.join(DIR_DM_BACKTEST,name,'hedged_returns.csv'),index_col=0,header=None).mean().values[0]

def summary():
    names = os.listdir(DIR_DM_BACKTEST)
    poolProcess=multiprocessing.Pool(10)
    rets=poolProcess.map(get_ret,names)
    s=pd.Series(rets,index=names).sort_values(ascending=False)
    s.name='mean_ret'
    s.to_csv(os.path.join(DIR_TMP,'data_mining_ret_summary.csv'))




if __name__ == '__main__':
    summary()



