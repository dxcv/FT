# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  14:54
# NAME:FT_hp-select_factors.py

import multiprocessing

from backtest_zht.main import run_backtest
from config import DIR_SIGNAL,DIR_RESULT_SPAN
import os
import pandas as pd

from singleFactor.combine.horse_race import get_spanning_signals

WINDOW=500 # trading day
NUM_PER_CATEGORY=2 # select the best n indicator in each category

def get_category(ind_name):
    #TODO: cluster the factors,rather than classify them mannually

    if ind_name.startswith('V__'):
        return 'V'
    elif ind_name.startswith('T__turnover'):
        return 'turnover'
    elif ind_name.startswith('T__mom'):
        return 'mom'
    elif ind_name.startswith(('T__vol','T__beta','T__idio')):
        return 'vol'
    elif ind_name.startswith('Q_'):
        return 'Q'
    elif ind_name.startswith('G_'):
        return 'G'
    elif ind_name.startswith('C_'):
        return 'C'
    else:
        raise ValueError('No category for : {}'.format(ind_name))

long_names=os.listdir(DIR_RESULT_SPAN)

_rets=[]
names=[]
for ln in long_names:
    try:
        _ret=pd.read_csv(os.path.join(DIR_RESULT_SPAN,ln,'hedged_returns.csv'),
                        index_col=0,parse_dates=True,header=None)
        _rets.append(_ret)
        names.append(ln)
    except:
        pass

ret=pd.concat(_rets,axis=1)
ret.columns=names

days=ret.iloc[:,0].resample('M').apply(lambda df:df.index[-1]).values#trick

days=days[20:] # prune the header

_get_cum_ret=lambda s:(1+s).cumprod().values[-1]-1

mss=[]

for day in days:
    sub=ret.loc[:day].last('2Y')
    ms=sub.apply(_get_cum_ret)
    mss.append(ms)
    print(day)

mark=pd.concat(mss,axis=1,keys=days)
mark=mark.stack().to_frame()
mark.index.names=['long_name','trd_dt']
mark.columns=['cum_ret']

mark['long_name']=mark.index.get_level_values('long_name')

mark['short_name']=mark['long_name'].map(lambda x:'__'.join(x.split('__')[:-1]))
mark['smooth']=mark['long_name'].map(lambda x:x.split('_')[-1])
mark['category']=mark['long_name'].map(get_category)

#select smooth window
selected=mark.groupby(['trd_dt','short_name']).apply(lambda df:df.loc[df['cum_ret'].idxmax()])

selected=selected.groupby(['trd_dt','category'],
    as_index=False,group_keys=False).apply(lambda df:df.nlargest(NUM_PER_CATEGORY,'cum_ret'))


