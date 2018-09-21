# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-20  20:39
# NAME:FT_hp-10 aggregation_with_fm.py

import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from data.dataApi import get_filtered_ret
# from empirical.chordia_and_yan.identify_anomalies1 import get_prominent_indicators, \
#     pricing_all_factors
# from empirical.config_ep import DIR_DM_NORMALIZED, DIR_CHORDIA
from empirical.config_ep import DIR_DM_GTA
from empirical.data_mining_gta.analyze_playing_field8 import \
    get_prominent_indicators
from empirical.data_mining_gta.dm_api import pricing_all_factors
from empirical.get_basedata import get_benchmark
from tools import multi_process


def fm_predict(indicators, smooth_period=None):
    '''
    get predicted stock return based on fama macbeth regression

    Args:
        indicators:DataFrame with multiIndex, does not include the 'const' column,

    Returns:DataFrame with multiIndex

    '''
    # ret = read_local('trading_m')['ret_m'].swaplevel()
    ret=get_filtered_ret().swaplevel()
    ind_inter = indicators.index.intersection(ret.index)
    indicators = indicators.reindex(ind_inter)
    ret = ret.reindex(ind_inter)

    indicator_1m = indicators.groupby('stkcd').shift(1)  # trick: use the indicators of time t-1
    comb = pd.concat([indicator_1m, ret], axis=1).dropna()

    comb = sm.add_constant(comb)
    indes = [col for col in comb.columns if col != 'ret_m']
    rs = comb.groupby('month_end').apply(
        lambda df: sm.OLS(df['ret_m'], df[indes]).fit())

    params = pd.concat([r.params for r in rs], axis=1, keys=rs.index).T

    ps=[]
    if smooth_period is None:
        for smooth_period in [1,6,12,36,60]:
            params = params.rolling(smooth_period).mean().dropna()
            indicators = indicators.loc[(params.index, slice(None)), :] #trick: do not use indicator_1m
            indicators = sm.add_constant(indicators)

            ss = []
            for month in params.index:
                sub = indicators.loc[(month, slice(None)), :]
                pred = np.array(sub) @ np.array(params.loc[month])
                s = pd.Series(pred, index=sub.index.get_level_values('stkcd'))
                s.name = month
                ss.append(s)
                print(month)

            predicted = pd.concat(ss, axis=1, sort=True).T.shift(1)  # trick: use the parameters and value of time t to predicted return in time t+1
            predicted = predicted.stack()
            predicted.name = smooth_period
            ps.append(predicted)
        ps=pd.concat(ps,axis=1)
        return ps
    else:
        params = params.rolling(smooth_period).mean().dropna()
        indicators = indicators.loc[(params.index, slice(None)),
                     :]  # trick: do not use indicator_1m
        indicators = sm.add_constant(indicators)

        ss = []
        for month in params.index:
            sub = indicators.loc[(month, slice(None)), :]
            pred = np.array(sub) @ np.array(params.loc[month])
            s = pd.Series(pred, index=sub.index.get_level_values('stkcd'))
            s.name = month
            ss.append(s)
            print(month)

        predicted = pd.concat(ss, axis=1, sort=True).T.shift(
            1)  # trick: use the parameters and value of time t to predicted return in time t+1
        predicted = predicted.stack()
        predicted.name = smooth_period
        return predicted


def tmb_with_fm_predicted(predicted):
    # ret = read_local('trading_m')['ret_m'].swaplevel()
    ret=get_filtered_ret().swaplevel()
    comb = pd.concat([predicted, ret], axis=1).dropna()
    comb.index.names = ['month_end', 'stkcd']

    comb['g'] = comb.groupby('month_end', group_keys=False).apply(
        lambda df: pd.qcut(df[predicted.name].rank(method='first'), 10,
                           labels=['g{}'.format(i) for i in range(1, 11)],
                           duplicates='raise'))

    port_ret_eq = comb.groupby(['month_end', 'g'])['ret_m'].mean().unstack(level=1)
    port_ret_eq.columns = port_ret_eq.columns.astype(str)

    tmb = port_ret_eq['g10'] - port_ret_eq['g1']
    return tmb

def pricing_with_fm_augmented(indicator_number):
    inds=get_prominent_indicators()
    selected= inds[:indicator_number]
    indicators=pd.concat([pd.read_pickle(os.path.join(DIR_DM_GTA,'normalized', sl + '.pkl')).stack() for sl in selected], axis=1, keys=selected)
    indicators=indicators.dropna(thresh=int(indicators.shape[1]*0.6))#trick: dropna and fillna with mean values
    # indicator=indicator.groupby('month_end').apply(lambda df:df.fillna(0))
    indicators=indicators.fillna(0)

    ps=fm_predict(indicators)
    ats=[]
    tmbs=[]
    for smooth_period in ps.columns:
        tmb=tmb_with_fm_predicted(ps[smooth_period])
        tmb.name=f'tmb_{smooth_period}_{indicator_number}'

        ff3=get_benchmark('ff3M')
        fmmodel=pd.concat([tmb, ff3], axis=1)

        at = pricing_all_factors(fmmodel)
        at.name= f'at_{smooth_period}_{indicator_number}'

        ats.append(at)
        tmbs.append(tmb)
    at_df=pd.concat(ats,axis=1,sort=True)
    tmb_df=pd.concat(tmbs,axis=1,sort=True)
    at_df.to_pickle(os.path.join(DIR_CHORDIA,f'at_fm_augmented_{indicator_number}.pkl'))
    tmb_df.to_pickle(os.path.join(DIR_CHORDIA,f'tmb_fm_augmented_{indicator_number}.pkl'))

def traverse_indicator_numbers():
    indicator_numbers = [1, 3, 5, 10, 15, 20, 26]
    for nb in indicator_numbers:
        pricing_with_fm_augmented(nb)
    multi_process(pricing_with_fm_augmented, indicator_numbers, n=7)


def analyze():
    indicator_numbers = [1, 3, 5, 10, 15, 20, 26]
    ss=[]
    for indicator_number in indicator_numbers:
        tmb_fm=pd.read_pickle(os.path.join(DIR_CHORDIA,f'tmb_fm_augmented_{indicator_number}.pkl'))[f'tmb_12_{indicator_number}']
        ss.append(tmb_fm)
    df=pd.concat(ss,axis=1)
    fig=df.cumsum().plot().get_figure()
    fig.savefig(os.path.join(DIR_CHORDIA,'tmb_fm_augmented.png'))

def main():
    traverse_indicator_numbers()
    analyze()

if __name__ == '__main__':
    main()



