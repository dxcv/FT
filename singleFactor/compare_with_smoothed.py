# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-17  08:55
# NAME:FT_hp-compare_with_smoothed.py
import multiprocessing
import pickle
from functools import partial
import matplotlib.pyplot as plt
from backtest.main import quick
from config import DIR_SIGNAL, DIR_SINGLE_BACKTEST, DIR_SIGNAL_SMOOTHED, \
    DIR_SIGNAL_PARAMETER, DIR_TMP
import os
import pandas as pd
from singleFactor.backtest_signal import SMOOTH_PERIODS, get_signal_direction


def find_best_smooth_period():
    names=os.listdir(os.path.join(DIR_SIGNAL_SMOOTHED,'0'))
    ret_df=pd.DataFrame(index=names,columns=SMOOTH_PERIODS,dtype=float)
    # turnover_df=pd.DataFrame(index=names,columns=SMOOTH_PERIODS,dtype=float)
    for name in names:
        for sp in SMOOTH_PERIODS:
            directory = os.path.join(DIR_SIGNAL_SMOOTHED, str(sp),name)
            try:
                avg_return=pd.read_csv(os.path.join(directory,'hedged_returns.csv'),index_col=0,parse_dates=True).mean()[0]
                # avg_turnover=pd.read_csv(os.path.join(directory,'turnover_rates.csv'),index_col=0,parse_dates=True).mean()[0]
                ret_df.at[name,sp]=avg_return
                # turnover_df.at[name,sp]=avg_turnover
            except:
                pass
        print(name)


    best_ret=ret_df.idxmax(axis=1).sort_values()
    # lowest_turnover=turnover_df.idxmin(axis=1).sort_values()
    return best_ret

def compare_parameters():
    names=os.listdir(os.path.join(DIR_SIGNAL_SMOOTHED,'0'))
    for name in names:
        ss_ret=[]
        # ss_turn=[]
        for sp in SMOOTH_PERIODS:
            directory = os.path.join(DIR_SIGNAL_SMOOTHED, str(sp),name)
            try:
                avg_return=pd.read_csv(os.path.join(directory,'hedged_returns.csv'),index_col=0,parse_dates=True,header=None)
                avg_return.columns=[str(sp)]
                ss_ret.append(avg_return)
                # avg_turnover=pd.read_csv(os.path.join(directory,'turnover_rates.csv'),index_col=0,parse_dates=True)
                # avg_turnover.name=sp
                # ss_turn.append(avg_turnover)
            except:
                pass
        df_ret=pd.concat(ss_ret,axis=1)
        (1+df_ret).cumprod().plot().get_figure().savefig(os.path.join(DIR_SIGNAL_PARAMETER,name+'.png'))
        # df_turn=pd.concat(ss_ret,axis=1)
        # (1+df_turn).cumprod().plot().get_figure().savefig(os.path.join(DIR_SIGNAL_PARAMETER,name+'.png'))
        print(name)

def get_return(args):
    name=args[0]
    sp=args[1]
    signal = pd.read_pickle(os.path.join(DIR_SIGNAL, name + '.pkl')) #trick: reverse the signal
    signal=signal*get_signal_direction(name)
    if sp:
        signal=signal.rolling(sp,min_periods=int(sp/2)).mean()
    results, fig = quick(signal, name, start='2010', end='2015')
    return sp,results['hedged_returns']

def _select_sp(name):
    pool=multiprocessing.Pool(16)
    ss=pool.map(get_return,((name,sp) for sp in range(0,101,2)))
    pool.close()
    pool.join()
    df=pd.concat([s[1] for s in ss],axis=1,keys=[s[0] for s in ss])

    # df.to_pickle(os.path.join(DIR_TMP,'df.pkl'))

    # df=pd.read_pickle(os.path.join(DIR_TMP,'df.pkl'))

    fig=df.mean().plot().get_figure()
    fig.savefig(os.path.join(DIR_TMP,name+'.png'))
    plt.close()

    # (1+df).cumprod().plot().get_figure().show()

def select_sp():
    names=[fn[:-4] for fn in os.listdir(DIR_SIGNAL)]
    for name in names:
        _select_sp(name)



# name = 'T__turnover2_std_300'
# get_return((name,1))



if __name__ == '__main__':
    # name = 'C__est_bookvalue_FT24M_to_close_g_20'
    select_sp()



#TODO： how to conduct papameter optimization?

