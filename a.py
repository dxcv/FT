# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-27  21:42
# NAME:FT_hp-a.py

import os
from itertools import combinations
import numpy as np

import pandas as pd
# import copy
import matplotlib.pyplot as plt
from config import DIR_TMP
from empirical.config_ep import DIR_BASEDATA

def hedged_year_performance(hedged_return):
    return_free = 0.03  # 无风险利率估算为3%

    year = pd.Series(hedged_return.index.year)
    year = year.drop_duplicates().values
    year = [str(y) for y in year]
    perf_year = pd.DataFrame(np.nan, index=year,
                             columns=['return',
                                      'max_drawdown',
                                      'volatility', 'sharp_ratio',
                                      'ret_draw_ratio'])
    for y in year:
        value = (hedged_return[y] + 1).cumprod()
        # bch_value = (benchmark_return[y] + 1).cumprod()

        max_down = 1 - min(value[y] / np.maximum.accumulate(value))
        volat = hedged_return[y].std() * (252 ** 0.5)

        day_number = len(value[y])
        if day_number > 230:
            ret = value[-1] - 1
            # bch_ret = bch_value[-1] - 1
        else:
            ret = value[-1] ** (252 / day_number) - 1
            # bch_ret = bch_value[-1] ** (252 / day_number) - 1
        # exs_ret = ret - bch_ret

        sharp = (ret - return_free) / volat
        ret_draw_ratio = ret / max_down
        perf_year.loc[y] = [ret, max_down, volat, sharp,
                            ret_draw_ratio]
    return perf_year

short_window='200_iw3_cw3_5_criteria3_150_1'
long_window='750_iw2_cw2_3_criteria3_150_2'

directory=r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_signal_backtest'

ret1=pd.read_csv(os.path.join(directory,short_window,'hedged_returns.csv'),index_col=0,header=None,parse_dates=True)
ret2=pd.read_csv(os.path.join(directory,long_window,'hedged_returns.csv'),index_col=0,header=None,parse_dates=True)

df=pd.concat([ret1,ret2],axis=1)
df['mean']=df.mean(axis=1)

((1+df).cumprod()-1).plot().get_figure().show()

perf1=hedged_year_performance(ret1.iloc[:,0])
perf2=hedged_year_performance(ret2.iloc[:,0])


perf=hedged_year_performance(df['mean'])

perf.to_csv(os.path.join(DIR_TMP,'perf.csv'))


# ret1=pd.read_csv(os.path.join(DIR_TMP,'rets1.csv'),index_col=0,parse_dates=True)['short_window_long_window']
# ret2=pd.read_csv(os.path.join(DIR_TMP,'rets1.csv'),index_col=0,parse_dates=True)['short_window_long_window']

# perf1=hedged_year_performance(ret1)
# perf2=hedged_year_performance(ret2)


