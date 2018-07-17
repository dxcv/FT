# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-14  23:02
# NAME:FT_hp-backtest_signal.py


import os
import pandas as pd
from backtest.main import quick

from config import DIR_DM_SIGNAL, DIR_DM_BACKTEST


def bt(name):
    signal=pd.read_pickle(os.path.join(DIR_DM_SIGNAL,name+'.pkl'))

    start,end= '2010','2015'
    results, fig = quick(signal, name, start=start,end=end)


    directory=os.path.join(DIR_DM_BACKTEST,name)
    os.makedirs(directory)
    fig.savefig(os.path.join(directory, name + '.png'))
    for k in results.keys():
        results[k].to_csv(os.path.join(directory, k + '.csv'))


fns=os.listdir(DIR_DM_SIGNAL)
names=[fn[:-4] for fn in fns]

for name in names:
    print(name)
    bt(name)



