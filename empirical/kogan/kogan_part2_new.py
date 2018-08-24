# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-15  14:09
# NAME:FT_hp-kogan_part2_new.py
import itertools
import statsmodels.api as sm
import numpy as np
from scipy import stats
import pandas as pd
from numba import jit

from line_profiler import LineProfiler

from empirical.bootstrap import bootstrap_kogan, get_data, pricing_assets
from tools import multi_process

CRITIC=0.05

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
    var_b = np.matrix(A).T @ np.matrix(MSE)
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    cal_pvalue = lambda t: 2 * (1 - stats.t.cdf(np.abs(t), X.shape[0] - 1))
    # pvalues = np.vectorize(cal_pvalue)(ts_b[0])# only get the pvalue of alpha
    # matched_num=(pvalues>CRITIC).sum()

    matched_num=sum([cal_pvalue(x)>CRITIC for x in ts_b[0]])

    return matched_num

def _for_one_combination(args):
    i,_names,simulated_factors=args
    model_factor_names = ['const', 'rp'] + list(_names)
    model = simulated_factors[model_factor_names].values
    assets = simulated_factors.drop(model_factor_names, axis=1).values
    print(i)
    return _names,get_matched_number(model,assets)

def observe_get_matched_number():
    ANOMALY_NUM = 100
    benchmark, raw_factors = get_data()
    realized_result = pricing_assets(benchmark, raw_factors)
    simulated_factors = bootstrap_kogan(benchmark, raw_factors, realized_result,
                                        ANOMALY_NUM)
    simulated_factors = sm.add_constant(simulated_factors)
    i=0
    _names=simulated_factors.columns[101:103]

    model_factor_names = ['const', 'rp'] + list(_names)
    model = np.matrix(simulated_factors[model_factor_names])
    assets = np.matrix(simulated_factors.drop(model_factor_names, axis=1))
    # model = simulated_factors[model_factor_names]
    # assets = simulated_factors.drop(model_factor_names, axis=1)

    # model.to_pickle(r'G:\FT_Users\HTZhang\empirical\model.pkl')
    # assets.to_pickle(r'G:\FT_Users\HTZhang\empirical\assets.pkl')

    lp=LineProfiler()
    lp_wrapper=lp(get_matched_number)
    lp_wrapper(model,assets)
    lp.print_stats()


def simulate_onetime():
    ANOMALY_NUM = 100
    benchmark, raw_factors = get_data()
    realized_result = pricing_assets(benchmark, raw_factors)
    simulated_factors = bootstrap_kogan(benchmark, raw_factors, realized_result,
                                        ANOMALY_NUM)
    simulated_factors = sm.add_constant(simulated_factors)

    def args_generator():
        i=0
        for _names in itertools.combinations([col for col in simulated_factors.columns if col not in ['const', 'rp']], 2):
            i+=1
            yield i,_names,simulated_factors

    results=multi_process(_for_one_combination, args_generator())
    _names_l=[r[0] for r in results]
    _matched_l=[r[1] for r in results]
    index=pd.MultiIndex.from_tuples(_names_l)
    matched_series=pd.Series(_matched_l,index=index)
    return matched_series

def test():
    ANOMALY_NUM = 100
    benchmark, raw_factors = get_data()
    realized_result = pricing_assets(benchmark, raw_factors)
    simulated_factors = bootstrap_kogan(benchmark, raw_factors, realized_result,
                                        ANOMALY_NUM)
    simulated_factors = sm.add_constant(simulated_factors)

    def args_generator():
        i = 0
        for _names in itertools.combinations(
                [col for col in simulated_factors.columns if
                 col not in ['const', 'rp']], 2):
            i += 1
            yield i, _names, simulated_factors

    # results = multi_process(_for_one_combination, args_generator())

    results=[_for_one_combination(args) for args in args_generator()]
    _names_l = [r[0] for r in results]
    _matched_l = [r[1] for r in results]
    index = pd.MultiIndex.from_tuples(_names_l)
    matched_series = pd.Series(_matched_l, index=index)
    return matched_series

# if __name__ == '__main__':
#
#     benchmark,raw_factors=get_data()
#
#     benchmark.to_pickle(r'G:\FT_Users\HTZhang\empirical\benchmark.pkl')
#     raw_factors.to_pickle(r'G:\FT_Users\HTZhang\empirical\raw_factors.pkl')


if __name__ == '__main__':
    import time
    t1=time.time()
    simulate_onetime()
    # test()
    print(time.time()-t1)

