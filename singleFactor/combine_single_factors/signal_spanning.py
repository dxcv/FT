# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-28  22:47
# NAME:FT_hp-signal_spanning.py
import multiprocessing
import os
import pandas as pd
# from backtest_zht.main import run_backtest
from backtest_zht.main_class import Backtest

from config import DIR_SIGNAL, DIR_BACKTEST_SPANNING, DIR_BACKTEST_SPANNING
# from singleFactor.backtest_signal import SMOOTH_PERIODS, get_smoothed_signal
from tools import multi_process


SMOOTH_PERIODS=[0,10,20,30,40,50,60,70,80]

def derive_signal(raw_signal, smooth, sign):
    raw_signal= raw_signal * sign
    if smooth==0:
        return raw_signal
    else:
        return raw_signal.rolling(smooth, min_periods=int(smooth / 2)).mean()

def get_derive_signal(short_name,smooth,sign):
    raw_signal=pd.read_pickle(os.path.join(DIR_SIGNAL,short_name+'.pkl'))
    return derive_signal(raw_signal,smooth,sign)

def _bt_one_set(fn):
    raw_signal = pd.read_pickle(os.path.join(DIR_SIGNAL, fn))
    for smooth in SMOOTH_PERIODS:
        for sign in [1, -1]:  # 1 denote positive, and -1 denotes negative
            name = '{}___smooth_{}___{}'.format(fn[:-4], smooth,
                                                {1: 'p', -1: 'n'}[sign])
            directory = os.path.join(DIR_BACKTEST_SPANNING, name)#fixme:
            signal=derive_signal(raw_signal,smooth,sign)
            Backtest(signal,name,directory)#fixme:
            # run_backtest(signal, name, directory)
            print(name)

def bt_all_spanning_signal():
    fns=os.listdir(DIR_SIGNAL)
    multi_process(_bt_one_set,fns,20,multi_paramters=False)
    # multiprocessing.Pool(20).map(_bt_one_set,fns)

def debug():
    fns=os.listdir(DIR_SIGNAL)
    for fn in fns:
        _bt_one_set(fn)

if __name__ == '__main__':
    bt_all_spanning_signal()
    # debug()
'''
employ stop loss strategy on the factors

Han, Y., Zhou, G., and Zhu, Y. (2016). Taming Momentum Crashes: A Simple Stop-Loss Strategy (Rochester, NY: Social Science Research Network).


'''


# debug: G__divdend3YR







