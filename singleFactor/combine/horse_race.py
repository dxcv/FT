# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-23  14:47
# NAME:FT_hp-horse_race.py
import multiprocessing

from backtest.main import quick
from config import DIR_HORSE_RACE
# from singleFactor.signal_spanning import get_all_signals
import os

WINDOW=4


import multiprocessing
from functools import partial

from backtest.main import quick
from config import DIR_SIGNAL,DIR_SIGNAL_SPAN
import os
import pandas as pd
from singleFactor.backtest_signal import SMOOTH_PERIODS, get_signal_direction, \
    get_smoothed_signal


def span(fn):
    signal=pd.read_pickle(os.path.join(DIR_SIGNAL,fn))
    signal=get_signal_direction(fn[:-4])*signal
    for sp in SMOOTH_PERIODS:
        new_signal=get_smoothed_signal(signal,sp)
        name='{}__sp_{}'.format(fn[:-4],sp)
        new_signal.to_pickle(os.path.join(DIR_SIGNAL_SPAN, name + '.pkl'))
    print(fn)

def get_spanning_signals(fn):
    '''

    Args:
        fn:

    Returns:tuple, (name,signal)

    '''
    signal = pd.read_pickle(os.path.join(DIR_SIGNAL, fn))
    signal = get_signal_direction(fn[:-4]) * signal
    for sp in SMOOTH_PERIODS:
        name='{}__sp_{}'.format(fn[:-4],sp)
        yield (name,get_smoothed_signal(signal, sp))

def task(args):
    end,fn=args
    for name,signal in get_spanning_signals(fn):
        start = end - WINDOW
        directory = os.path.join(DIR_HORSE_RACE, str(end), name)

        if os.path.exists(directory):
            pass
        else:
            os.makedirs(directory)

        results, fig = quick(signal, fig_title=name, start=str(start),
                             end=str(end))
        fig.savefig(os.path.join(directory, name + '.png'))
        for k in results.keys():
            results[k].to_csv(os.path.join(directory, k + '.csv'))

def get_args_list():
    fns=os.listdir(DIR_SIGNAL)
    args_list=[]
    for end in range(2007,2019):
        for fn in fns:
            args_list.append((end,fn))
    return args_list

def run():
    multiprocessing.Pool(10).map(task,get_args_list())


if __name__ == '__main__':
    run()



