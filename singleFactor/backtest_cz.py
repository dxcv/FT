# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-07  17:11
# NAME:FT-backtest_cz.py
import multiprocessing

from config import SINGLE_D_INDICATOR, DIR_SIGNAL, DIR_BACKTEST_FIG
import os
import pandas as pd
from data.dataApi import read_local
from backtest.main import quick


# name='C__est_price_90_relative.pkl'

def indicator_to_signal(name):
    df=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))
    # mould_index=read_local('mould_index')
    fdmt=read_local('equity_fundamental_info')
    comb=pd.concat([df,fdmt],axis=1)
    comb=comb.sort_index()
    comb=comb.ffill(limit=31)
    signal=comb.reindex(fdmt.index)[name]
    signal=signal.unstack().T
    signal.to_pickle(os.path.join(DIR_SIGNAL,name+'.pkl'))
    print(name)

def convert_all():
    path = r'D:\app\python36\zht\internship\FT\singleFactor\indicators.xlsx'
    df = pd.read_excel(path, sheet_name='valid')
    names=df['name']

    pool = multiprocessing.Pool(4)
    pool.map(indicator_to_signal,names)

    # for i,name in enumerate(names):
    #     indicator_to_signal(name)
    #     print(i,name)


def test_one(fn):
    signal = pd.read_pickle(os.path.join(DIR_SIGNAL, fn))
    results, fig = quick(start=signal.index[0], end=signal.index[-1],
                         signal=signal, mark_title='test')
    fig.savefig(os.path.join(DIR_BACKTEST_FIG, fn[:-4] + '.png'))
    print(fn)


def test_all():
    fns=os.listdir(DIR_SIGNAL)
    pool = multiprocessing.Pool(4)
    pool.map(test_one,fns)

if __name__ == '__main__':
    convert_all()
    test_all()





