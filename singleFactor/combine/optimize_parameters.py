# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-28  09:20
# NAME:FT_hp-optimize_parameters.py

import multiprocessing

from backtest_zht.main import run_backtest
from config import DIR_SIGNAL, DIR_RESULT_SPAN, DIR_TMP
import os
import pandas as pd
from pandas.tseries.offsets import MonthEnd,Day
from singleFactor.backtest_signal import get_signal

from singleFactor.combine.horse_race import get_spanning_signals
from singleFactor.combine.rolling_combination import get_mixed_signals

'''
1. 指标的切换频率由每年调整变为每月调整
2. window,rolling expanding
3. 每类指标个数
4. 调整指标之间的加权方式以及大类之间的加权方式
5. 2009年开始回测
6. 选用不同的评分指标
7. effective number 由200变为100
8. cluster to get category rather than manually
9. PCA machine learning
10. 每天剔除失效的策略,加入新的策略，就如同选股股票调仓的方式，比如定义回撤超过20为失效的策略。
11. stop loss strategy
    Han, Y., Zhou, G., and Zhu, Y. (2016). Taming Momentum Crashes: A Simple Stop-Loss Strategy (Rochester, NY: Social Science Research Network).


'''
FREQ='M'
WINDOW=500 # trading day
NUM_PER_CATEGORY=5 # select the best n indicator in each category
MIX_TYPE='equal'
RATING_METHOD='cum_ret'
EFFECT_NUM=200


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

def get_strategy_ret():
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
    ret.to_pickle(os.path.join(DIR_TMP,'ret.pkl'))

    return ret


_get_cum_ret=lambda s:(1+s).cumprod().values[-1]-1

def rating_strategies(ret,rating_func=_get_cum_ret):
    days=ret.iloc[:,0].resample(FREQ).apply(lambda df:df.index[-1]).values#trick
    days=days[20:] # prune the header


    mss=[]

    for day in days:
        sub=ret.loc[:day].last(WINDOW)
        #fixme: all the strategies should have been tested for at least 2 years and make sure that all the strategies share the same history window.
        sub=sub.dropna(axis=1,thresh=int(sub.shape[0]*0.95))
        ms=sub.apply(rating_func)
        mss.append(ms)
        print(day)

    rating=pd.concat(mss,axis=1,keys=days)
    rating=rating.stack().to_frame()
    rating.index.names=['long_name','trd_dt']
    rating.columns=['cum_ret']
    rating['long_name']=rating.index.get_level_values('long_name')
    rating['short_name']=rating['long_name'].map(lambda x:'__'.join(x.split('__')[:-1]))
    rating['smooth']=rating['long_name'].map(lambda x:x.split('_')[-1])
    rating['category']=rating['long_name'].map(get_category)
    rating.to_pickle(os.path.join(DIR_TMP,'rating.pkl'))

    return rating

def select_strategies(rating):
    #select the best smooth window
    selected=rating.groupby(['trd_dt','short_name'],as_index=False,group_keys=False).apply(lambda df:df.loc[df[RATING_METHOD].idxmax()])

    #TODO: filter the negative strategies. For example, there are some strategies who have a negative cum_ret.
    selected=selected.groupby(['trd_dt','category'],
        as_index=False,group_keys=False).apply(lambda df:df.nlargest(NUM_PER_CATEGORY,RATING_METHOD))
    selected.drop('short_name',axis=1,inplace=True)
    table=selected.reset_index()

    table.to_pickle(os.path.join(DIR_TMP,'table.pkl'))
    return table


def generate_signal(table):
    trd_dts = table['trd_dt'].unique()
    # table['next_month'] = pd.to_datetime(table['trd_dt']) + MonthEnd(0) + Day(1)
    # table['next_month'] = table['next_month'].map(lambda x: x.strftime('%Y-%m'))

    signal_monthly = []
    # for trd_dt in trd_dts[:-1]:  # fixme:
    for i in range(len(trd_dts[:-1])):
        rb_dt=trd_dts[i]
        next_rb_dt=trd_dts[i+1]
        sub = table[table['trd_dt'] == rb_dt]
        y_signal_l = []
        for c in sub['category'].unique():
            ss = sub[sub['category'] == c]
            c_signal_l = []
            for _, row in ss.iterrows():
                name = '__'.join(row.loc['long_name'].split('__')[:-1])
                sp = row.loc['long_name'].split('_')[-1]
                c_signal_l.append(get_signal(name,sp)[rb_dt:next_rb_dt][1:])#trick:do not include the trd_dt
                # next_month = row.loc['next_month']
                # c_signal_l.append(get_signal(name, sp)[next_month])
            c_signal = get_mixed_signals(c_signal_l)
            y_signal_l.append(c_signal)
        y_signal = get_mixed_signals(y_signal_l)
        signal_monthly.append(y_signal)
        print(rb_dt)

    comb_signal = pd.concat(signal_monthly)

    run_backtest(comb_signal, 'test', os.path.join(DIR_TMP, 'test_comb_signal'),
                 start='2009')


