# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-04  09:44
# NAME:FT-master-simulation.py
import itertools
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import random
import matplotlib.pyplot as plt

from empirical.config import DIR_KOGAN, CRITICAL, DIR_KOGAN_RESULT
from empirical.get_basedata import get_benchmark
from empirical.replication import get_raw_factors, match_based_on_alpha_pvalue
from tools import multi_task

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

def simulate_one_time(args):
    realized_params, betas, alpha_stderr, anomaly_num=args
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
    
    evaluate the performace based on the p-value of alpha, different with the empirical
    part where GRS is used.
     
    '''
    _matched_l=[]
    _names_l=[]
    for _names in list(itertools.combinations(sim_factors.columns,2)):
        #'const' serve as the intercept in regression
        sim_model=pd.concat([draw_params[['const','rp']],sim_factors[list(_names)]],axis=1)
        matched_num=0
        for _fn in sim_factors.columns:
            if _fn not in _names:
                Y=sim_factors[_fn]
                r=sm.OLS(Y,sim_model).fit()
                p=r.pvalues.loc['const']# p value of the alpha
                if p>CRITICAL:
                    matched_num+=1
        _matched_l.append(matched_num)
        _names_l.append(_names)
    index=pd.MultiIndex.from_tuples(_names_l)
    matched=pd.Series(_matched_l,index=index)
    # matched=pd.DataFrame(_matched_l,columns=['model_factors','matched'])
    return matched
    # matched['matched'].plot.kde().get_figure().show()



def gen_args_list(realized_params,betas,alpha_stderr,anomaly_num,n):
    for i in range(n):
        yield (realized_params,betas,alpha_stderr,anomaly_num)

def get_kde(sim_num=100,anomaly_numbers=(0,5,10,15,20)):
    '''Figure 5: factor model performance histogram

    for visulization methods, refer to
        https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        https://pythonhosted.org/PyQt-Fit/KDE_tut.html
    '''
    raw_factors=get_raw_factors()
    # bench_names = ['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M', 'ff6M'] #TODO: add pca model
    benchmark = get_benchmark('ff3M')#fixme: set pca_modela s benchmark
    betas,alpha_stderr,realized_params=get_estimated_coefs(raw_factors,benchmark)

    result_l=[]
    for anomaly_num in anomaly_numbers:
        print(anomaly_num)
        args_generator=gen_args_list(realized_params,betas,alpha_stderr,anomaly_num,sim_num)
        _result=pd.concat(multi_task(simulate_one_time, args_generator), axis=1).stack()
        result_l.append(_result)
        # a=mcs.mean(axis=1).sort_values() #TODO: analysis whether the matched number for each model is stable. Some model may always match more factors

    result=pd.concat(result_l,axis=1,keys=anomaly_numbers)
    result.to_csv(os.path.join(DIR_KOGAN_RESULT,'frequency_{}.csv'.format(sim_num)))
    result.plot.kde(bw_method=0.3).get_figure().savefig(os.path.join(DIR_KOGAN_RESULT,'kde_{}.png'.format(sim_num)))

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
    _result=pd.concat(multi_task(simulate_one_time,args_generator),axis=1)
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
    get_fig6()

# if __name__ == '__main__':
#     for n in [10,30,70,100]:
#         get_kde(sim_num=n)

