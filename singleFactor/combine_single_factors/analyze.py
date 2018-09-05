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


# short='150_iw3_cw2_10_criteria3_150_3'
short_window='200_iw3_cw3_5_criteria3_150_1'
medium_window='500_iw2_cw2_10_criteria3_150_2'
long_window='750_iw2_cw2_3_criteria3_150_2'

directory=r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_signal_backtest'
sets=['short_window','long_window']


combs=list(combinations(sets,1))+list(combinations(sets,2))+list(combinations(sets,3))


rets=[]
for comb in combs:
    name='_'.join(comb)
    ss=[]
    for c in comb:
        s = pd.read_csv(os.path.join(directory, eval(c), 'hedged_returns.csv'),
                        index_col=0, header=None).iloc[:, 0]
        # s.name = c
        ss.append(s)
    df=pd.concat(ss,axis=1)
    comret=((1+df).cumprod().mean(axis=1)-1)
    comret.index=pd.to_datetime(comret.index)
    comret.name=name
    rets.append(comret)

rets=pd.concat(rets,axis=1)
rets.plot().get_figure().show()



ss=[]
for c in [short_window,long_window]:
    s = pd.read_csv(os.path.join(directory, c, 'hedged_returns.csv'),
                    index_col=0, header=None).iloc[:, 0]
    ss.append(s)
pd.concat(ss,axis=1).corr()