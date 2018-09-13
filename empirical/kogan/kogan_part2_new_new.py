# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-23  17:02
# NAME:FT_hp-kogan_part2_new_new.py

import itertools
import multiprocessing
import os
import random

import numpy as np
from empirical.bootstrap import bootstrap_kogan, pricing_assets
from empirical.config_ep import CRITICAL, DIR_KOGAN, DIR_EP
from scipy import stats
import pandas as pd
from numba import jit

import statsmodels.api as sm
from tools import multi_process


def get_matched_number(X,Y):
    '''
    use X to pricing Y, and count the number of asset that can be matched by X
    Args:
        X: array, X is benchmark.values, and the first column of X must be 'const'
        Y: array, Y is assets.values, the assets to be priced

    Returns:

    '''
    '''
    the firs column of model must be 'const'
    Args:
        model:
        assets:
        add_constant:

    Returns:

    '''
    params = np.linalg.pinv(X.T @ X) @ X.T @ Y
    predictions = X @ params

    #trick:split the the long equation can speed up signicantly, 10 time faster with this way
    # a=(Y-predictions).T
    # b=(Y-predictions)
    # c=a@b # use matrix rather than array ,or it will be 10 time slower
    # d=c.diagonal()
    # MSE=d/(X.shape[0]-X.shape[1])

    MSE = ((Y - predictions).T @ (Y - predictions)).diagonal() / (X.shape[0] - X.shape[1])
    A = np.linalg.inv(X.T @ X).diagonal()
    var_b=A.T @ MSE
    sd_b = np.sqrt(var_b)#fixme: standard error if not standard deviation
    ts_b = params / sd_b
    cal_pvalue = lambda t: 2 * (1 - stats.t.cdf(np.abs(t), X.shape[0] - 1))

    # matched_num=0
    # for x in np.nditer(ts_b[0]):
    #     if cal_pvalue(x)>CRITIC:
    #         matched_num+=1
    #
    pvalues = np.vectorize(cal_pvalue)(ts_b[0])# only get the pvalue of alpha
    matched_num=(pvalues>CRITICAL).sum()

    return matched_num

def _for_one_combination(args):
    _names,simulated_factors=args
    model_factor_names = ['const', 'rp'] + list(_names)
    model = np.matrix(simulated_factors[model_factor_names])#trick: matrix
    assets = np.matrix(simulated_factors.drop(model_factor_names, axis=1)) #trick: matrix
    return _names,get_matched_number(model,assets)

def get_data():
    benchmark=pd.read_pickle(os.path.join(DIR_KOGAN,'benchmark.pkl'))
    raw_factors=pd.read_pickle(os.path.join(DIR_KOGAN,'raw_factors.pkl'))
    return benchmark,raw_factors

def match_with_all_possible_three_factor_models(factors):
    '''
    :param factors:DataFrame, factors must contain two columns named ['const','rp']
    :return: MultiIndex Series
    '''
    def args_generator():
        for _names in itertools.combinations([col for col  in factors.columns if col not in ['const', 'rp']], 2):
            yield _names,factors
    results=multi_process(_for_one_combination,args_generator(),6)
    _names_l=[r[0] for r in results]
    _matched_l=[r[1] for r in results]
    index=pd.MultiIndex.from_tuples(_names_l)
    matched_series=pd.Series(_matched_l,index=index)
    return matched_series

def get_realized_result():
    benchmark, raw_factors = get_data()
    raw_factors['rp']=benchmark['rp']
    raw_factors['const']=1
    realized=match_with_all_possible_three_factor_models(raw_factors)
    realized.to_pickle(os.path.join(DIR_KOGAN,'realized.pkl'))


def simulate_onetime(_id,benchmark,raw_factors,realized_result,anomaly_num):
    simulated_factors = bootstrap_kogan(benchmark, raw_factors, realized_result,
                                        anomaly_num)
    simulated_factors = sm.add_constant(simulated_factors)
    return match_with_all_possible_three_factor_models(simulated_factors)


def simulate(sim_num=10,anomaly_num=0):
    benchmark, raw_factors = get_data()
    realized_result = pricing_assets(benchmark, raw_factors)
    ss=[simulate_onetime(i,benchmark,raw_factors,realized_result,anomaly_num) for i in range(sim_num)]
    df=pd.concat(ss,axis=1)
    df.to_pickle(os.path.join(DIR_KOGAN,f'{anomaly_num}_{sim_num}.pkl'))
    # df.to_pickle(r'E:\tmp_kogan\{}_{}.pkl'.format(anomaly_num,sim_num))

def run():
    for anomaly_num in [60,110]:#fixme:
        simulate(sim_num=100,anomaly_num=anomaly_num)
        print(anomaly_num)

def get_fig5():
    '''Fig5'''
    ans=[0, 10, 50,60,100,110, 150, 194]
    ss=[]
    for an in ans:
        # s=pd.read_pickle(r'e:\tmp_kogan\{}_100.pkl'.format(an)).stack()
        df=pd.read_pickle(os.path.join(DIR_KOGAN,f'{an}_100.pkl'))
        s = df.apply(lambda s: s.value_counts() / len(s)).reindex(index=range(1, 194)).fillna(0).mean(axis=1)
        s.name=an
        ss.append(s)
        print(an)
    realized = pd.read_pickle(os.path.join(DIR_KOGAN, 'realized.pkl'))
    realized=realized.value_counts()/len(realized)
    realized=realized.reindex(index=range(194))
    realized.name='realized'
    comb=pd.concat(ss+[realized],axis=1)
    # comb.plot.kde(bw_method=0.3).get_figure().savefig(os.path.join(DIR,'distribution_all.pdf'))
    comb.plot().get_figure().savefig(os.path.join(DIR_KOGAN,'distribution_all.pdf'))

