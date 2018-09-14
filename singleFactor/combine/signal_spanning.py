# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-23  14:52
# NAME:FT_hp-backtest_spanning_signal.py

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

def get_all_signals():
    fns = os.listdir(DIR_SIGNAL)
    for fn in fns:
        yield from get_spanning_signals(fn)

def span_all():
    fns = os.listdir(DIR_SIGNAL)
    multiprocessing.Pool(10).map(span,fns)


if __name__ == '__main__':
    span_all()
