# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-18  09:46
# NAME:FT_hp-combine.py
from singleFactor.compare_with_smoothed import find_best_smooth_period
import multiprocessing
import pickle
from functools import partial
import matplotlib.pyplot as plt
from backtest.main import quick
from config import DIR_SIGNAL, DIR_SINGLE_BACKTEST, DIR_SIGNAL_SMOOTHED, \
    DIR_SIGNAL_PARAMETER, DIR_TMP, DIR_SIGNAL_COMB
import os
import numpy as np
import pandas as pd
from singleFactor.backtest_signal import SMOOTH_PERIODS, get_signal_direction, \
    get_smoothed_signal
from tools import outlier, z_score

names = os.listdir(os.path.join(DIR_SIGNAL_SMOOTHED, '0'))
ret_df = pd.DataFrame(index=names, columns=SMOOTH_PERIODS, dtype=float)


def traverse_one_sp(sp):
    fns=os.listdir(os.path.join(DIR_SIGNAL_SMOOTHED,str(sp)))
    get_sharpe=lambda fn:pd.read_csv(os.path.join(DIR_SIGNAL_SMOOTHED,str(sp),fn,'hedged_perf.csv'),index_col=0,header=None).loc['sharp_ratio'].values[0]
    return pd.Series([get_sharpe(fn) for fn in fns],index=fns)

def standardize_signal(signal):
    '''
    Args:
        signal:DataFrame, panel

    Returns:DataFrame, panel,the shape may be different with the input dataframe

    '''
    stk=signal.stack()
    stk=stk.groupby('trd_dt').apply(outlier)
    stk=stk.groupby('trd_dt').apply(z_score)
    return stk.unstack()


def select_with_sharpe(thresh=1.0):
    '''

    Args:
        thresh:least sharpe value to be selected

    Returns:DataFrame,with two columns,['sahrpe','sp']

    '''
    ss=multiprocessing.Pool(10).map(traverse_one_sp,SMOOTH_PERIODS)
    sharpe_info=pd.concat(ss,axis=1,keys=SMOOTH_PERIODS)

    # sharpe_info = pd.read_csv(os.path.join(DIR_SIGNAL_COMB, 'sharpe_info.csv'),
    #                           index_col=0)
    sp = sharpe_info[sharpe_info > thresh].idxmax(axis=1).sort_values().dropna()
    sharpe = sharpe_info.loc[sp.index].max(axis=1)
    result = pd.concat([sp, sharpe], axis=1, keys=['sp', 'sharpe']).sort_values(
        'sp', ascending=False)

    result.to_csv(os.path.join(DIR_SIGNAL_COMB, 'selected_indicators.csv'))




def get_outer_frame(dflist):
    indsets=[set(df.index.tolist()) for df in dflist]
    colsets=[set(df.columns.tolist()) for df in dflist]
    indOuter=sorted(list(set.union(*indsets)))
    colOuter=sorted(list(set.union(*colsets)))
    return [df.reindex(index=indOuter,columns=colOuter) for df in dflist]


def get_mixed_signal():
    manually_selcted=pd.read_csv(os.path.join(DIR_SIGNAL_COMB,'manually_selected.csv'),index_col=0)
    manually_selcted=manually_selcted.dropna()

    for c in manually_selcted['manually_selected'].unique():
        subdf=manually_selcted[manually_selcted['manually_selected']==c]
        signals=[]
        for name,row in subdf.iterrows():
            sp=row['sp']
            signal=pd.read_pickle(os.path.join(DIR_SIGNAL,name+'.pkl'))*get_signal_direction(name)
            signal=standardize_signal(signal)#trick: standardize the signal before aggregation
            if sp:
                signal=get_smoothed_signal(signal,sp)
                # signal=signal.rolling(sp,min_periods=int(sp/2)).mean()
            # signals.append(signal.stack())
            signals.append(signal)
            print(c,name)

        signals=get_outer_frame(signals)
        #TODO: standardized before aggregation
        mixed=pd.DataFrame(np.nanmean([s.values for s in signals],axis=0),index=signals[0].index,columns=signals[0].columns)

        mixed.to_pickle(os.path.join(DIR_SIGNAL_COMB,'mixed_signal',c+'.pkl'))



def backtest_one(fn):
    name=fn[:-4]
    directory=os.path.join(DIR_SIGNAL_COMB,'combine',name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        return

    signal=pd.read_pickle(os.path.join(DIR_SIGNAL_COMB,'mixed_signal',name+'.pkl'))
    results,fig=quick(signal,name,start='2010')
    fig.savefig(os.path.join(directory, name + '.png'))
    for k in results.keys():
        results[k].to_csv(os.path.join(directory, k + '.csv'))


def backtest_mixed_mixed(selected=True):
    fns=os.listdir(os.path.join(DIR_SIGNAL_COMB,'mixed_signal'))
    if selected:
        fns=[fn for fn in fns if fn[:-4] in ['C','Q','V']]
        name='cqv'
    else:
        name='mixed'
    signals=[]
    for fn in fns:
        signal=pd.read_pickle((os.path.join(DIR_SIGNAL_COMB,'mixed_signal',fn)))
        signal=standardize_signal(signal)
        signals.append(signal)
        print(fn)
    signals=get_outer_frame(signals)
    mixed = pd.DataFrame(np.nanmean([s.values for s in signals], axis=0),
                         index=signals[0].index, columns=signals[0].columns)

    directory=os.path.join(DIR_SIGNAL_COMB,'combine',name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        return

    results,fig=quick(mixed,name,start='2010')
    fig.savefig(os.path.join(directory, '{}.png'.format(name)))
    for k in results.keys():
        results[k].to_csv(os.path.join(directory, k + '.csv'))


if __name__ == '__main__':
    # select_with_sharpe()

    # get_mixed_signal()
    # fns = os.listdir(os.path.join(DIR_SIGNAL_COMB, 'mixed_signal'))
    # multiprocessing.Pool(5).map(backtest_one,fns)

    backtest_mixed_mixed()
