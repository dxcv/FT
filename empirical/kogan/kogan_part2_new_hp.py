# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-01  22:27
# NAME:zht-base_on_foxit.py
import itertools
import multiprocessing
import random
import matplotlib.pyplot as plt

import numpy as np
from empirical.bootstrap import get_data
from scipy import stats
from line_profiler import LineProfiler
import pandas as pd
from numba import jit
import os
import time
import statsmodels.api as sm


CRITIC=0.05

DIR_RESULT=r'G:\FT_Users\HTZhang\empirical\kogan\results'


def multi_task(func, args_iter, n=4):
    pool=multiprocessing.Pool(n)
    results=pool.map(func, args_iter)
    pool.close()#trick: close the processing every time the pool has finished its task, and pool.close() must be called before pool.join()
    pool.join()
    #refer to https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    return results


def pricing_assets(benchmark, assets):
    '''
    use benchmark to pricing the assets and get the relevant parameters including
    betas,alpha stderr, resid
    Args:
        benchmark:
        assets: DataFrame, each column represent a realized asset

    Returns:

    '''
    # E:\FT_Users\HTZhang\software\python\HTZhang\FT_hp\empirical\kogan_part2.py
    X=sm.add_constant(benchmark)
    rs=[sm.OLS(assets[col],X).fit() for col in assets.columns]

    alpha=pd.Series([r.params['const'] for r in rs], index=assets.columns)
    alpha_stderr=pd.Series([r.bse['const'] for r in rs], index=assets.columns)
    alpha_t=pd.Series([r.tvalues['const'] for r in rs], index=assets.columns)
    alpha_p=pd.Series([r.pvalues['const'] for r in rs],index=assets.columns)
    betas=pd.concat([r.params[1:] for r in rs], axis=1, keys=assets.columns)
    resid=pd.DataFrame([r.resid for r in rs], index=assets.columns, columns=benchmark.index).T

    result={}
    for _name in ['alpha','alpha_stderr','alpha_t','alpha_p','betas','resid']:
        result[_name]=eval(_name)
    return result

def bootstrap_kogan(benchmark, assets, realized_result, anomaly_num):
    get_random_index = lambda: random.choices(benchmark.index,
                                              k=benchmark.shape[0])

    _01_list=[1]*anomaly_num+[0]*(len(realized_result['alpha'])-anomaly_num)
    random.shuffle(_01_list)
    is_anomaly=pd.Series(_01_list,index=realized_result['alpha'].index)
    assigned_alpha=realized_result['alpha_stderr'].map(lambda a:3*a*random.choice([-1,1]))*is_anomaly
    adjusted_assets=assets-realized_result['alpha']+assigned_alpha
    index=get_random_index()
    simulated_factors=adjusted_assets.loc[index]
    simulated_rp=benchmark['rp'].loc[index]
    simulated_factors=pd.concat([simulated_factors,simulated_rp],axis=1)
    return simulated_factors



@jit(nopython=True)
def get_matched_number_jit(X,Y):
    params = np.linalg.pinv(X.T @ X) @ X.T @ Y
    predictions = X @ params
    MSE=np.diag(np.transpose(Y-predictions) @ (Y-predictions))/(X.shape[0]-X.shape[1])
    A=np.diag(np.linalg.inv(np.transpose(X) @ X))
    # var_b=np.diag(np.matrix(A)) @ np.matrix(MSE)
    var_b=np.diag(A) @ MSE
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    # cal_pvalue = lambda t: 2 * (1 - stats.t.cdf(np.abs(t), X.shape[0] - 1))
    matched_num=0
    for x in np.nditer(ts_b[0]):
        v=2*(1-stats.t.cdf(np.abs(x),X.shape[0]-1)) #fixme: failed, numba does not support stats.t
        if v>CRITIC:
            matched_num+=1

    return matched_num

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
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    cal_pvalue = lambda t: 2 * (1 - stats.t.cdf(np.abs(t), X.shape[0] - 1))

    # matched_num=0
    # for x in np.nditer(ts_b[0]):
    #     if cal_pvalue(x)>CRITIC:
    #         matched_num+=1
    #
    pvalues = np.vectorize(cal_pvalue)(ts_b[0])# only get the pvalue of alpha
    matched_num=(pvalues>CRITIC).sum()

    return matched_num

def _for_one_combination(args):
    i,_names,simulated_factors=args
    model_factor_names = ['const', 'rp'] + list(_names)
    model = np.matrix(simulated_factors[model_factor_names])#trick: matrix
    assets = np.matrix(simulated_factors.drop(model_factor_names, axis=1)) #trick: matrix
    # print(i)
    return _names,get_matched_number(model,assets)

# def get_data():
#     benchmark=pd.read_pickle(r'E:\tmp_kogan\benchmark.pkl')
#     raw_factors=pd.read_pickle(r'E:\tmp_kogan\raw_factors.pkl')
#     return benchmark,raw_factors

def simulate_onetime(_id,benchmark,raw_factors,realized_result,anomaly_num):
    simulated_factors = bootstrap_kogan(benchmark, raw_factors, realized_result,
                                        anomaly_num)
    simulated_factors = sm.add_constant(simulated_factors)
    def args_generator():
        i=0
        for _names in itertools.combinations([col for col in simulated_factors.columns if col not in ['const', 'rp']], 2):
            i+=1
            yield i,_names,simulated_factors

    # results=multi_process(_for_one_combination,args_generator(),8)#fixme:
    results=[_for_one_combination(args) for args in args_generator()]
    _names_l=[r[0] for r in results]
    _matched_l=[r[1] for r in results]
    index=pd.MultiIndex.from_tuples(_names_l)
    matched_series=pd.Series(_matched_l,index=index)
    print(_id)
    return matched_series

def simulate(sim_num=10,anomaly_num=0):
    benchmark, raw_factors = get_data()
    realized_result = pricing_assets(benchmark, raw_factors)
    ss=[simulate_onetime(i,benchmark,raw_factors,realized_result,anomaly_num) for i in range(sim_num)]
    df=pd.concat(ss,axis=1)
    df.to_pickle(os.path.join(DIR_RESULT,'{}_{}.pkl'.format(anomaly_num,sim_num)))
    # df.to_pickle(r'E:\tmp_kogan\{}_{}.pkl'.format(anomaly_num,sim_num))

def run():
    # for anomaly_num in [0,10,50,100,150,194]:
    #     simulate(sim_num=1,anomaly_num=anomaly_num)
    #     print(anomaly_num)

    for sim_num in [1,10]:
        t1=time.time()
        simulate(sim_num,anomaly_num=0)
        print(sim_num,time.time()-t1)

def analyze_with_barchart():

    df = pd.read_pickle(r'e:\a\df.pkl')


    counts = df.stack().value_counts().reindex(range(200))
    plt.figure(figsize=(20, 8))
    counts.plot.bar()
    plt.savefig(r'e:\a\sampled_10_100.pdf')


def analyze():
    ans=[0, 10, 50, 100, 150, 194]
    df=pd.concat([pd.read_pickle(r'e:\tmp_kogan\{}_1.pkl'.format(an)) for an in ans],axis=1,keys=ans)
    df.plot.kde(bw_method=0.3).get_figure().savefig(r'e:\tmp_kogan\distribution.pdf')

if __name__ == '__main__':
    run()






def profiler_get_matched():
    benchmark,raw_factors=get_data()
    benchmark,raw_factors=np.matrix(benchmark),np.matrix(raw_factors)
    lp=LineProfiler()
    lp_wrapper=lp(get_matched_number)
    lp_wrapper(benchmark,raw_factors)
    # get_matched_number(benchmark,raw_factors)
    lp.print_stats()

def profiler_simulate_onetime():
    ANOMALY_NUM = 100
    benchmark, raw_factors = get_data()
    realized_result = pricing_assets(benchmark, raw_factors)
    lp = LineProfiler()
    lp_wrapper = lp(simulate_onetime)
    lp_wrapper(0,benchmark,raw_factors,realized_result,ANOMALY_NUM)
    # get_matched_number(benchmark,raw_factors)
    lp.print_stats()

def profiler_for_one_combination():
    ANOMALY_NUM = 100
    benchmark, raw_factors = get_data()
    realized_result = pricing_assets(benchmark, raw_factors)
    simulated_factors = bootstrap_kogan(benchmark, raw_factors, realized_result,
                                        ANOMALY_NUM)
    simulated_factors = sm.add_constant(simulated_factors)

    # def args_generator():
    #     i=0
    #     for _names in itertools.combinations([col for col in simulated_factors.columns if col not in ['const', 'rp']], 2):
    #         i+=1
    #         yield i,_names,simulated_factors

    i=0
    _names=simulated_factors.columns[100:102]

    args=(i, _names, simulated_factors)
    lp = LineProfiler()
    lp_wrapper = lp(_for_one_combination)
    lp_wrapper(args)
    # get_matched_number(benchmark,raw_factors)
    lp.print_stats()


def debug():
    benchmark, raw_factors = get_data()
    benchmark, raw_factors = np.matrix(benchmark), np.matrix(raw_factors)
    get_matched_number(benchmark,raw_factors)


# if __name__ == '__main__':
#     profiler_simulate_onetime()
    # profiler_for_one_combination()


