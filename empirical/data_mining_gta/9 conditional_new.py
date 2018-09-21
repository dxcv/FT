# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-20  20:09
# NAME:FT_hp-9 conditional_new.py
from empirical.config_ep import DIR_DM_GTA, CROSS_LEAST
import pandas as pd
import os

from empirical.data_mining_gta.dm_api import get_playing_indicators
from tools import multi_process

G=5

def conditional_alpha(args):
    ind,cond_variable=args
    path=os.path.join(DIR_DM_GTA,'analyse','conditional','cache',f'{cond_variable}__{ind}.pkl')
    if os.path.exists(path):
        return

    indicator= pd.read_pickle(os.path.join(DIR_DM_GTA,'normalized',ind+'.pkl')).shift(1).stack() #trick use the indicator an conditional variable in time t-1
    conditional = pd.read_pickle(os.path.join(DIR_DM_GTA, 'conditional', cond_variable + '.pkl')).shift(1).stack()
    ret = pd.read_pickle(os.path.join(DIR_DM_GTA, 'fdmt_m.pkl'))['ret_m']
    comb = pd.concat([indicator, ret, conditional], axis=1)
    comb.columns=[ind,'ret_m',cond_variable]
    comb.index.names = ['month_end', 'stkcd']
    comb = comb.dropna(subset=['ret_m', cond_variable])
    comb = comb.groupby('month_end').filter(lambda df: df.shape[
                                                           0] > CROSS_LEAST)  # trick: filter out months with too small sample
    comb = comb.fillna(0)

    #groupby conditional variable
    comb['gc'] = comb.groupby('month_end', group_keys=False).apply(
        lambda df: pd.qcut(df[cond_variable].rank(method='first'), 5,
                           labels=[f'g{i}' for i in range(1, 6)]))

    # groupby factor
    comb['gf'] = comb.groupby(['month_end', 'gc'], group_keys=False).apply(
        lambda df: pd.qcut(df[ind].rank(method='first'),G,
                           labels=[f'g{i}' for i in range(1, G+1)]))

    stk = comb.groupby(['month_end', 'gc', 'gf']).apply(
        lambda df: df['ret_m'].mean()).unstack('gf')
    panel = (stk[f'g{G}'] - stk['g1']).unstack()
    panel.columns = panel.columns.astype(str)
    panel['all'] = panel.mean(axis=1)
    panel['high-low'] = panel['g5'] - panel['g1']

    alpha = panel.mean()
    t = panel.mean() / panel.sem()  # trick: tvalue = mean / stderr,   stderr = std / sqrt(n-1) ,pd.Series.sem() = pd.Series.std()/pow(len(series),0.5)

    table = pd.concat([alpha, t], axis=1, keys=['alpha', 't']).T
    table.to_pickle(path)
    print(cond_variable,ind)

def main():
    playing_indicators=get_playing_indicators()
    fns = os.listdir(os.path.join(DIR_DM_GTA, 'conditional'))

    conds = [fn[:-4] for fn in fns]
    header=['log_size','turnover_20','idio_6M']
    conds=header+[co for co in conds if co not in header]

    for cond in conds:
        args_generator=((ind,cond) for ind in playing_indicators)
        multi_process(conditional_alpha,args_generator,n=10,size_in_each_group=100)

if __name__ == '__main__':
    main()


