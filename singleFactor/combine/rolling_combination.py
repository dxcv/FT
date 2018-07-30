# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-24  09:20
# NAME:FT_hp-rolling_combination.py
import multiprocessing

from config import DIR_HORSE_RACE, DIR_SIGNAL, DIR_TMP
import os
import pandas as pd
from singleFactor.backtest_signal import get_signal_direction, get_signal, \
    run_backtest
from singleFactor.combine.combine import standardize_signal, get_outer_frame
import numpy as np

'''
1. 指标的切换频率由每年调整变为每月调整
2. window,rolling expanding
3. 每类指标个数
4. 调整指标之间的加权方式以及大类之间的加权方式
5. 2009年开始回测
6. 选用不同的评分指标
7. effective number 由200变为100
'''

PARAMS={
    'freq':'M',
    'top_num':2,
    'baseon':'sharpe',#return_down_ration
    'effective_num':100,
    'signal_to_weight_mode':3,
}


def get_category(ind_name):
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

def _get_sharpe(args):
    year,ind_name=args
    fp=os.path.join(DIR_HORSE_RACE,year,ind_name,'hedged_perf.csv')
    v=pd.read_csv(fp, index_col=0, header=None).loc['sharp_ratio'].values[0]
    return year,ind_name,v

def get_sharps():
    args_list=[]
    # years=os.listdir(DIR_HORSE_RACE)
    years=[str(i) for i in range(2009,2018)]
    for year in years:
        indnames=os.listdir(os.path.join(DIR_HORSE_RACE,year))
        for indname in indnames:
            args_list.append((year,indname))
    items=multiprocessing.Pool(32).map(_get_sharpe,args_list)
    df=pd.DataFrame(items,columns=['year','long_name','sharpe'])
    df['short_name'] = df['long_name'].map(lambda s: s.split('sp')[0][:-2])
    df['smooth'] = df['long_name'].map(lambda s: s.split('sp')[-1][1:])
    # df = df.set_index(['year', 'short_name', 'smooth'])
    # df.to_pickle(os.path.join(DIR_HORSE_RACE, 'sharpe.pkl'))
    return df

def select_based_on_sharpe(top=5):
    # sharpe=pd.read_pickle(os.path.join(DIR_HORSE_RACE,'sharpe.pkl'))
    sharpe=get_sharps()
    #select smooth window
    sharpe=sharpe.groupby(['year','short_name']).apply(lambda df:df.loc[df['sharpe'].idxmax()])
    sharpe=sharpe.reset_index(drop=True)
    sharpe['category']=sharpe['long_name'].map(get_category)

    #trick:only choose from the top 80%
    sharpe=sharpe.groupby('year',as_index=False,group_keys=False).apply(lambda df:df.nlargest(int(df.shape[0]*0.8),'sharpe'))

    #trick:only select the top 5
    sharpe=sharpe.groupby(['year','category'],as_index=False,group_keys=False).apply(lambda df:df.nlargest(top,'sharpe'))
    return sharpe

def get_mixed_signals(signals):
    signals=[standardize_signal(signal) for signal in signals]
    signals = get_outer_frame(signals)
    mixed = pd.DataFrame(np.nanmean([s.values for s in signals], axis=0),
                         index=signals[0].index, columns=signals[0].columns)
    return mixed

def task(args):
    sharpe,year=args
    subdf = sharpe[sharpe['year'] == year]
    y_signal_l = []
    for category in subdf['category'].unique():
        print(year, category)
        ssdf = subdf[subdf['category'] == category]
        c_signal_l = []
        for _, row in ssdf.iterrows():
            name = '__'.join(row.loc['long_name'].split('__')[:-1])
            sp = row.loc['long_name'].split('_')[-1]
            c_signal_l.append(get_signal(name, sp)[str(int(year)+1)])
        c_signal = get_mixed_signals(c_signal_l)
        y_signal_l.append(c_signal)
    y_signal = get_mixed_signals(y_signal_l)
    return  y_signal

def get_args_list(sharpe):
    args_list=[]
    for year in sorted(sharpe['year'].unique()):
        args_list.append((sharpe,year))
    return args_list


def combine_signal():
    args_list=get_args_list(select_based_on_sharpe())
    signal_fragments=multiprocessing.Pool(10).map(task,args_list)
    comb=pd.concat(signal_fragments)
    run_backtest(comb, 'test', os.path.join(DIR_TMP, 'test'), start='2010')


# if __name__ == '__main__':
    # get_sharps()
    # combine_signal()



