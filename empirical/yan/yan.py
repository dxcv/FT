# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-14  10:55
# NAME:FT_hp-yan.py
import os
import random

import pandas as pd
import statsmodels.api as sm
from empirical.get_basedata import get_benchmark
from empirical.kogan.kogan_part1 import get_raw_factors
from scipy import stats
from tools import multi_process

BENCH='ff3M'

DIR_RESULT=r'G:\FT_Users\HTZhang\empirical\yan'

def get_data():
    benchmark = get_benchmark(BENCH)
    raw_factors = get_raw_factors()
    base_index=benchmark.index.intersection(raw_factors.index)
    #trick: unify the index
    return benchmark.reindex(base_index),raw_factors.reindex(base_index)

# def get_realized_params(benchmark,raw_factors):
#     # E:\FT_Users\HTZhang\software\python\HTZhang\FT_hp\empirical\kogan_part2.py
#     comb = pd.concat([benchmark, raw_factors], axis=1).dropna()
#     resids_l = []
#     betas_l = []
#     stderr_l = []
#     for factorName in raw_factors.columns:
#         Y = comb[factorName]
#         X = comb[benchmark.columns]
#         X = sm.add_constant(X)
#         r = sm.OLS(Y, X).fit()
#         resids_l.append(r.resid)
#         betas_l.append(r.params[1:])
#         stderr_l.append(r.bse['const'])
#     betas = pd.concat(betas_l, axis=1, keys=raw_factors.columns).T
#     alpha_stderr = pd.Series(stderr_l, index=raw_factors.columns)
#     resid = pd.DataFrame(resids_l, index=raw_factors.columns, columns=comb.index).T
#     return betas,alpha_stderr,resid
#
# def _get_t_alpha(benchmark, factors):
#     ts = []
#     alphas = []
#     X = sm.add_constant(benchmark)
#     for ps in factors.columns:
#         Y = factors[ps]
#         r = sm.OLS(Y, X).fit()
#         ts.append(r.tvalues['const'])
#         alphas.append(r.params['const'])
#
#     ts = pd.Series(ts, index=factors.columns)
#     alphas = pd.Series(alphas, index=factors.columns)
#     return ts,alphas

def bootstrap_onetime(args):
    i,raw_factors,benchmark,resid,betas=args
    sampled_index = random.choices(raw_factors.index, k=raw_factors.shape[0])
    sampled_bc=benchmark.loc[sampled_index]
    sampled_resid=resid.loc[sampled_index]
    fixed_term=pd.DataFrame(sampled_bc.values@betas.T.values,index=sampled_bc.index,columns=betas.index)
    psuedo_factors=fixed_term+sampled_resid #zore alpha
    #pricing the psuedo_factors
    ts,alphas=_get_t_alpha(sampled_bc, psuedo_factors)
    print(i)
    return ts,alphas

def bootstrap_N(N=100):
    benchmark,raw_factors=get_data()

    betas,alpha_stderr,resid=get_realized_params(benchmark,raw_factors)
    args_generator=((i,raw_factors,benchmark,resid,betas) for i in range(N))
    # results=[bootstrap_onetime(args) for args in args_generator]
    results=multi_process(bootstrap_onetime, args_generator)
    sample_tvalue_df=pd.concat([r[0] for r in results],axis=1)
    sample_alpha_df=pd.concat([r[1] for r in results],axis=1)
    sample_alpha_df.to_pickle(os.path.join(DIR_RESULT,'sample_alpha_df.pkl'))
    sample_tvalue_df.to_pickle(os.path.join(DIR_RESULT,'sample_tvalue_df.pkl'))
    return sample_tvalue_df,sample_alpha_df

def get_realized_result():
    benchmark,raw_factors=get_data()
    realized_t,realized_alpha=_get_t_alpha(benchmark, raw_factors)


benchmark,raw_factors=get_data()
sample_alpha_df=pd.read_pickle(os.path.join(DIR_RESULT,'sample_alpha_df.pkl'))
sample_tvalue_df=pd.read_pickle(os.path.join(DIR_RESULT,'sample_tvalue_df.pkl'))
realized_t, realized_alpha = _get_t_alpha(benchmark, raw_factors)



# pers=list(range(11))+list(range(90,101))
pers=range(101)
xalphas=[]
palphas=[]

xts=[]
pts=[]
for i in pers:
    qalpha=sample_alpha_df.quantile(i/100,axis=1)
    _xalpha=realized_alpha.quantile(i/100)
    _palpha=stats.percentileofscore(qalpha,_xalpha)/100
    xalphas.append(_xalpha)
    palphas.append(_palpha)

    qt=sample_tvalue_df.quantile(i/100,axis=1)
    _xt=realized_t.quantile(i/100)
    _pt=stats.percentileofscore(qt,_xt)/100
    xts.append(_xt)
    pts.append(_pt)


result_alpha=pd.DataFrame(list(zip(xalphas,palphas,xts,pts)),index=pers,columns=['realized_alpha','pvalue_alpha','realized_t','pvalue_t'])








# if __name__ == '__main__':
#     bootstrap_N()












