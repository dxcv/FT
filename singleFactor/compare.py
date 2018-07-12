# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-12  16:14
# NAME:FT_hp-compare.py
import os
import pandas as pd
from backtest.main import quick
from config import SINGLE_D_INDICATOR, DIR_BACKTEST_RESULT, LEAST_CROSS_SAMPLE
from data.dataApi import read_local
from singleFactor.check import daily_to_monthly, check_factor
from tools import clean

name = 'C__est_bookvalue_FT24M_to_close_g_20'


def test1():

    directory = os.path.join(DIR_BACKTEST_RESULT, name)

    signal=pd.read_csv(os.path.join(directory,'signal.csv'),index_col=0,parse_dates=True)
    tmp=pd.DataFrame(signal.index,index=signal.index)
    trd_dt=tmp.resample('M').last()
    signal_monthly=signal.resample('M').last()
    signal_monthly.index=trd_dt.index

    signal_monthly=signal_monthly.shift(1)
    signal_monthly=signal_monthly.reindex(signal.index)
    signal_monthly=signal_monthly.ffill(limit=31)

    results,fig=quick(signal_monthly,fig_title='test',start='2010')

    #TODO：span the index

    fig.show()


directory = os.path.join(DIR_BACKTEST_RESULT, name)
signal = pd.read_csv(os.path.join(directory, 'signal.csv'), index_col=0,
                     parse_dates=True)
signal=signal.shift(-1)
signal_m=signal.resample('M').last().stack().to_frame()


