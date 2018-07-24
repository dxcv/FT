# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-20  17:12
# NAME:FT_hp-select_model_every_year.py


import multiprocessing
import os
from functools import partial

import pandas as pd
from backtest.main import quick

from config import DIR_DM_SIGNAL, DIR_DM_BACKTEST, DIR_DM_BACKTEST_LONG, \
    DIR_DM_RACE



def bt1(name,start,end):
    if end is None:
        directory=os.path.join(DIR_DM_RACE,'latest',name)
    else:
        directory=os.path.join(DIR_DM_RACE,end,name)

    if os.path.exists(directory):
        return # skip
    else:
        os.makedirs(directory)

    signal=pd.read_pickle(os.path.join(DIR_DM_SIGNAL,name+'.pkl'))
    results,fig = quick(signal, name, start=start, end=end)
    if results['hedged_returns'].sum() < 0:
        signal = -signal
        results, fig = quick(signal, name, start=start, end=end)

    fig.savefig(os.path.join(directory, name + '.png'))
    for k in results.keys():
        results[k].to_csv(os.path.join(directory, k + '.csv'))

def task(args):
    name=args[0]
    start=args[1]
    end=args[2]
    bt1(name,start,end)

def main():
    fns=os.listdir(DIR_DM_SIGNAL)
    names = [fn[:-4] for fn in fns]

    arglist=[]
    for name in names:
        for end in ['2010','2011','2012','2013','2014','2015','2016','2017',None]:
            if end is None:
                start = '2013'
            elif int(end)<=2012:
                start='2008'
            else:
                start=str(int(end)-4)
            arglist.append((name,start,end))

    pool=multiprocessing.Pool(20)
    pool.map(task,arglist)



#TODO: only backtest with the sample covering 2017-today to find the best strategy that can outperform zz500.

if __name__ == '__main__':
    main()
