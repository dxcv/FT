# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  14:33
# NAME:FT_hp-backtest_spanning_signals.py


import multiprocessing

import os

from backtest_zht.main import quick, run_backtest
from singleFactor.combine.horse_race import get_spanning_signals
from config import DIR_SIGNAL,DIR_RESULT_SPAN

def backtest(signal, name, directory,start=None,end=None):
    if os.path.exists(directory) and len(os.listdir(directory))>0:
        return # skip
    elif not os.path.exists(directory):
        os.makedirs(directory)

    results,fig=quick(signal,name,start=start,end=end)
    results['hedged_returns'].to_csv(os.path.join(directory,'hedged_returns.csv'))
    #
    #
    # fig.savefig(os.path.join(directory,name+'.png'))
    # for k in results.keys():
    #     results[k].to_csv(os.path.join(directory,k+'.csv'))



def do_one_fn(fn):
    for name,signal in get_spanning_signals(fn):
        directory = os.path.join(DIR_RESULT_SPAN,name)
        run_backtest(signal,name,directory)

def do_all():
    fns = os.listdir(DIR_SIGNAL)
    multiprocessing.Pool(15).map(do_one_fn,fns)


if __name__ == '__main__':
    do_all()
