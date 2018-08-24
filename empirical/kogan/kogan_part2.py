# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-04  09:44
# NAME:FT-master-kogan_part2.py
import itertools
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from empirical.config_ep import CRITICAL, DIR_KOGAN_RESULT
from empirical.get_basedata import get_benchmark
from empirical.kogan.kogan_part1 import get_raw_factors, \
    match_based_on_alpha_pvalue
from tools import multi_process

DIR_TMP=r'G:\FT_Users\HTZhang\FT\tmp'

def get_estimated_coefs(raw_factors,benchmark):
    comb=pd.concat([benchmark, raw_factors], axis=1).dropna()
    resids_l=[]
    betas_l=[]
    stderr_l=[]
    for factorName in raw_factors.columns:
        Y=comb[factorName]
        X=comb[benchmark.columns]
        X=sm.add_constant(X)
        r=sm.OLS(Y,X).fit()
        resids_l.append(r.resid)
        betas_l.append(r.params[1:])
        stderr_l.append(r.bse['const'])
    betas=pd.concat(betas_l,axis=1,keys=raw_factors.columns).T
    alpha_stderr=pd.Series(stderr_l,index=raw_factors.columns)
    resid=pd.DataFrame(resids_l,index=raw_factors.columns,columns=comb.index).T
    realized_params=pd.concat([benchmark, resid], axis=1).dropna()
    realized_params=sm.add_constant(realized_params)
    return betas,alpha_stderr,realized_params

def _get_matched_num(args):
    sim_factors,draw_params,_names=args
    # 'const' serve as the intercept in regression
    sim_model = pd.concat(
        [draw_params[['const', 'rp']], sim_factors[list(_names)]], axis=1)
    fns = [fn for fn in sim_factors.columns if fn not in _names]
    matched_num=0
    for _fn in fns:  # factor names
        Y = sim_factors[_fn]
        r = sm.OLS(Y, sim_model).fit()
        p = r.pvalues.loc['const']  # p value of the alpha
        if p > CRITICAL:
            matched_num += 1
    return matched_num,_names

def simulate_one_time(args):
    i,realized_params, betas, alpha_stderr, anomaly_num=args
    is_anomaly_l=[True]*anomaly_num+[False]*(betas.shape[0]-anomaly_num)
    random.shuffle(is_anomaly_l)
    is_anomaly_d=dict(zip(betas.index,is_anomaly_l))

    draw_params=realized_params.sample(n=realized_params.shape[0], replace=True)#resample time index
    draw_params.index=realized_params.index

    sim_factors_l=[]
    for factorName in betas.index:
        coefs=betas.loc[factorName]
        # simulated alpha
        coefs['const']= alpha_stderr[factorName] * 3 * random.choice([-1, 1]) if is_anomaly_d[factorName] else 0
        coefs[factorName]=1
        sub=draw_params[coefs.index.tolist()]
        sim_factor=(sub*coefs).sum(axis=1)
        sim_factors_l.append(sim_factor)

    #simulated factors
    sim_factors=pd.concat(sim_factors_l,axis=1,keys=betas.index)
    '''
    test each possible three-factor model's performance, consisting of the simulated
    market portfolio and two simulated factors among our 21 simulated return factors.
    
    evaluate the performace based on the p-value of alpha (time series regression), different with the empirical
    part where GRS is used ( panel regression).

    '''
    args_generator=((sim_factors,draw_params,_names) for _names in itertools.combinations(sim_factors.columns,2))
    result=multi_process(_get_matched_num, args_generator)
    _matched_l=[r[0] for r in result]
    _names_l=[r[1] for r in result]
    index=pd.MultiIndex.from_tuples(_names_l)
    matched_series=pd.Series(_matched_l,index=index)
    print(i)
    return matched_series

def gen_args_list(realized_params,betas,alpha_stderr,anomaly_num,n):
    for i in range(n):
        yield (i,realized_params,betas,alpha_stderr,anomaly_num)

def get_sampled_result(sim_num=100, anomaly_number=10):
    '''Figure 5: factor model performance histogram

    for visulization methods, refer to
        https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        https://pythonhosted.org/PyQt-Fit/KDE_tut.html
    '''
    raw_factors=get_raw_factors()
    # bench_names = ['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M', 'ff6M'] #TODO: add pca model
    benchmark = get_benchmark('ff3M')#fixme: set pca_model as benchmark
    betas,alpha_stderr,realized_params=get_estimated_coefs(raw_factors,benchmark)

    args_generator = gen_args_list(realized_params, betas, alpha_stderr,
                                   anomaly_number, sim_num)
    matched_ss=[]
    for args in args_generator:
        matched_ss.append(simulate_one_time(args))
    sampled_result=pd.concat(matched_ss,axis=1)
    sampled_result.to_pickle(os.path.join(DIR_KOGAN_RESULT,'sampled_match_{}_{}.pkl'.format(anomaly_number,sim_num)))
    # TODO: analysis whether the matched number for each model is stable. Some model may always match more factors

    # result.plot.kde(bw_method=0.3).get_figure().savefig(os.path.join(DIR_KOGAN_RESULT,'kde_{}.png'.format(sim_num)))

def analyze_10_10():
    sampled_result=pd.read_pickle(os.path.join(DIR_KOGAN_RESULT,'sampled_match_10_10.pkl'))
    # sampled_result.stack().plot.kde(bw_method=0.3)
    # plt.show()

    # sampled_result.stack().hist(bins=200)
    # plt.show()
    counts=sampled_result.stack().value_counts().reindex(range(200))
    plt.figure(figsize=(20,8))
    counts.plot.bar()
    plt.savefig(os.path.join(DIR_KOGAN_RESULT,'sampled_10_10.pdf'))


def sampling_distribution(anomaly_num,sim_num=100):
    '''
    Fig 6: Factor model performance histogram -sampling distribution
    Args:
        anomaly_num:
        sim_num:

    Returns:

    '''
    raw_factors=get_raw_factors()
    # bench_names = ['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M', 'ff6M'] #TODO: add pca model
    benchmark = get_benchmark('ff3M')#fixme: set pca_modela s benchmark
    betas,alpha_stderr,realized_params=get_estimated_coefs(raw_factors,benchmark)
    args_generator = gen_args_list(realized_params, betas, alpha_stderr,
                                   anomaly_num, sim_num)
    _result=pd.concat(multi_process(simulate_one_time, args_generator), axis=1)
    _result.to_pickle(os.path.join(DIR_TMP,'_result.pkl'))


    _result=pd.read_pickle(os.path.join(DIR_TMP,'_result.pkl'))

    sample=_result.apply(lambda s:s.value_counts())
    sample= sample / sample.sum()
    sample=sample.reindex(range(22)).fillna(0)#fixme: N+1

    qs=sample.quantile([0.05,0.5,0.95], axis=1).T

    historical=match_based_on_alpha_pvalue()
    historical=historical.value_counts()
    historical/=historical.sum()
    historical=historical.reindex(range(22)).fillna(0)#fixme: N+1
    historical.plot().get_figure().show()
    historical.name='historical'
    pd.concat([qs,historical],axis=1).plot()
    plt.savefig(os.path.join(DIR_KOGAN_RESULT,'fig6_{}.png'.format(anomaly_num)))
    plt.close()

def get_fig6():
    for n in range(3,8):
        sampling_distribution(n)
        print(n)


if __name__ == '__main__':
    get_sampled_result(sim_num=10,anomaly_number=10)





'''
Ideas:
1. bootstrap with the original factor returns to get the distribution of the matched number for each benchmark


#TODO: compare the distribution of the number of anomalies based on different benchmarks

'''