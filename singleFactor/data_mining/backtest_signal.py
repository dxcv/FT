# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-14  23:02
# NAME:FT_hp-backtest_signal.py
import multiprocessing
import os
from functools import partial

import pandas as pd
from backtest.main import quick

from config import DIR_DM_SIGNAL, DIR_DM_BACKTEST, DIR_DM_BACKTEST_LONG


def bt(name,long=False):
    print(name)
    if long:
        directory=os.path.join(DIR_DM_BACKTEST_LONG,name)
        start,end='2010',None
    else:
        directory=os.path.join(DIR_DM_BACKTEST,name)
        start,end= '2010','2015'
    if os.path.exists(directory):
        return # skip
    else:
        os.makedirs(directory)

    signal=pd.read_pickle(os.path.join(DIR_DM_SIGNAL,name+'.pkl'))

    results, fig = quick(signal, name, start=start,end=end)
    if results['hedged_returns'].sum()<0:
        signal=-signal
        results, fig = quick(signal, name, start=start,end=end)
    fig.savefig(os.path.join(directory, name + '.png'))
    for k in results.keys():
        results[k].to_csv(os.path.join(directory, k + '.csv'))

def test_all(long=False):
    fns=os.listdir(DIR_DM_SIGNAL)
    names=[fn[:-4] for fn in fns]
    pool=multiprocessing.Pool(16)
    pool.map(partial(bt,long=long),names)

if __name__ == '__main__':
    test_all(long=True)


