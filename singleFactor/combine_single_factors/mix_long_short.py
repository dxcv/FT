# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-05  09:18
# NAME:FT_hp-get_prominent_indicators.py


import os
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt
from backtest_zht.main_class import DEFAULT_CONFIG, Backtest
from config import DIR_MIXED_SIGNAL, DIR_TMP
from singleFactor.combine_single_factors.combine_new import standardize_signal


def average_plot():
    short_window='200_iw3_cw3_5_criteria3_150_1'
    medium_window='500_iw2_cw2_10_criteria3_150_2'
    long_window='750_iw2_cw2_3_criteria3_150_2'

    directory=r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_signal_backtest'
    sets=['short_window','long_window']


    combs=list(combinations(sets,1))+list(combinations(sets,2))+list(combinations(sets,3))


    rets1=[]
    rets2=[]
    for comb in combs:
        name='_'.join(comb)
        ss=[]
        for c in comb:
            s = pd.read_csv(os.path.join(directory, eval(c), 'hedged_returns.csv'),
                            index_col=0, header=None).iloc[:, 0]
            # s.name = c
            ss.append(s)
        df=pd.concat(ss,axis=1)

        comret1=((1+df).cumprod().mean(axis=1)-1)
        comret1.index=pd.to_datetime(comret1.index)
        comret1.name=name

        comret2=((1+df.mean(axis=1)).cumprod()-1)
        comret2.index=pd.to_datetime(comret2.index)
        comret2.name=name

        rets1.append(comret1)
        rets2.append(comret2)

    rets1=pd.concat(rets1,axis=1)
    rets1.plot().get_figure().show()

    rets2=pd.concat(rets2,axis=1)
    rets2.plot().get_figure().show()

    rets1.to_csv(os.path.join(DIR_TMP,'rets1.csv'))
    rets2.to_csv(os.path.join(DIR_TMP,'rets2.csv'))



    # ss=[]
    # for c in [short_window,long_window]:
    #     s = pd.read_csv(os.path.join(directory, c, 'hedged_returns.csv'),
    #                     index_col=0, header=None).iloc[:, 0]
    #     ss.append(s)
    # pd.concat(ss,axis=1).corr()


def average_signal():
    short_window='200_iw3_cw3_5_criteria3'
    long_window='750_iw2_cw2_3_criteria3'
    signal1=pd.read_pickle(os.path.join(DIR_MIXED_SIGNAL,short_window+'.pkl'))
    signal2=pd.read_pickle(os.path.join(DIR_MIXED_SIGNAL,long_window+'.pkl'))

    signal=standardize_signal(signal1)+standardize_signal(signal2)

    effective_number=150
    signal_weight_mode=2
    name='average_signal'
    directory=os.path.join(DIR_TMP,name)

    cfg = DEFAULT_CONFIG
    cfg['effective_number'] = effective_number
    cfg['signal_to_weight_mode'] = signal_weight_mode
    Backtest(signal, name=name, directory=directory, start='2009',
             config=cfg)  # TODO: start='2009'

def average_portfolio():
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

    short_window = '200_iw3_cw3_5_criteria3_150_1'
    long_window = '750_iw2_cw2_3_criteria3_150_2'

    directory = r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_signal_backtest'

    ret1 = pd.read_csv(
        os.path.join(directory, short_window, 'hedged_returns.csv'),
        index_col=0, header=None, parse_dates=True)
    ret2 = pd.read_csv(
        os.path.join(directory, long_window, 'hedged_returns.csv'), index_col=0,
        header=None, parse_dates=True)

    df = pd.concat([ret1, ret2], axis=1)
    df['mean'] = df.mean(axis=1)

    ((1 + df).cumprod() - 1).plot().get_figure().show()

    perf1 = hedged_year_performance(ret1.iloc[:, 0])
    perf2 = hedged_year_performance(ret2.iloc[:, 0])

    perf = hedged_year_performance(df['mean'])

    perf.to_csv(os.path.join(DIR_TMP, 'perf.csv'))


    # if __name__ == '__main__':
#     average_the_portfolio()