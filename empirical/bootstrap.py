# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-15  10:52
# NAME:FT_hp-bootstrap.py
import random

import pandas as pd
import statsmodels.api as sm
from empirical.get_basedata import get_benchmark, get_data


# from empirical.kogan.kogan_part1 import get_raw_factors


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


def bootstrap_residual_independently(benchmark,assets,realized_result):
    get_random_index=lambda:random.choices(benchmark.index,k=benchmark.shape[0])

    #bootstrap residual independently for each asset and use the realized fixed term
    '''
    resample residual (independently)
    
    1. only resample residual, and use realized benchmark and factors.
    
    resample residual independently for each asset.
    '''
    fixed_term=pd.DataFrame(benchmark.values@realized_result['betas'].values,columns=realized_result['betas'].columns)
    # resample the time index independently for each asset
    sampled_resid=pd.concat([realized_result['resid'][col].loc[get_random_index()].reset_index(drop=True)
                             for col in realized_result['resid'].columns],
                            axis=1,keys=realized_result['resid'].columns)
    pseudo_assets=fixed_term+sampled_resid
    pseudo_assets.index=assets.index

    re=pricing_assets(benchmark,pseudo_assets)

def bootstrap_residual_and_factor(benchmark,assets,realized_result):
    get_random_index = lambda: random.choices(benchmark.index,
                                              k=benchmark.shape[0])
    '''
    resample residual (independently)
    resample fixed term (dependently)
    
    2. resample both the residuals and factor returns and resample them independently. 
    When resampling factor returns, we use the same draw across all funds 
    (to preserve the correlation of factor returns on all funds).
    '''
    sampled_benchmark=benchmark.loc[get_random_index()]
    fixed_term=pd.DataFrame(sampled_benchmark.values@realized_result['betas'].values,columns=realized_result['betas'].columns)
    sampled_resid=pd.concat([realized_result['resid'][col].loc[get_random_index()].reset_index(drop=True)
                             for col in realized_result['resid'].columns],
                            axis=1,keys=realized_result['resid'].columns)
    pseudo_assets=fixed_term+sampled_resid
    pseudo_assets.index=assets.index

    re=pricing_assets(benchmark,pseudo_assets)

def bootstrap_cross_sectional(benchmark,assets,realized_result):
    get_random_index = lambda: random.choices(benchmark.index,
                                              k=benchmark.shape[0])
    '''
    resample residual (dependently)
    
    In this procedure, rather than drawing sequences of time periods t_i that are unique 
    to each fund i, we draw T time periods from the set { t=1,...,T} and then 
    resample residuals for this reindexed time sequence across all funds, 
    thereby preserving any cross-sectional correlation in the residuals.'''
    fixed_term=pd.DataFrame(benchmark.values@realized_result['betas'].values,columns=realized_result['betas'].columns)
    sampled_resid=realized_result['resid'].loc[get_random_index()].reset_index(drop=True)
    pseudo_assets=fixed_term+sampled_resid
    pseudo_assets.index=assets.index

    re=pricing_assets(benchmark,pseudo_assets)

def bootstrap_ff(benchmark,assets,realized_result):
    get_random_index = lambda: random.choices(benchmark.index,
                                              k=benchmark.shape[0])
    '''
    
    substract the estimated alpha from the original data to get the psuedo
    asset and then bootstrap the time index ( that is, all the assets share 
    the same random sample of time index)
    '''
    adjusted_assets=assets-realized_result['alpha']
    sampled_index=get_random_index()
    pseudo_assets=adjusted_assets.loc[sampled_index]
    pseudo_benchmark=benchmark.loc[sampled_index]
    pseudo_assets.reset_index(drop=True,inplace=True) #trick:handle the problem of duplicated index
    pseudo_benchmark.reset_index(drop=True,inplace=True)
    re=pricing_assets(pseudo_benchmark,pseudo_assets)
    return re

def bootstrap_both_factor_and_model(benchmark,assets):
    '''
    Do not change the structure of the original data, just resample from the original data.

    The only difference with bootstrap_ff is that this method to not substract the 'alpha'
    from the assets.

    Args:
        benchmark:
        assets:
    Returns:

    '''
    random_index=random.choice(benchmark.index,k=benchmark.shape[0])
    sp_benchmark=benchmark.loc[random_index]
    sp_assets=assets.loc[random_index]
    return sp_benchmark,sp_assets

def bootstrap_kogan(benchmark, assets, realized_result, anomaly_num):
    '''
    just like bootstrap_ff, but you can assign an anomaly_num for the assets
    Args:
        benchmark:
        assets:
        realized_result:
        anomaly_num:

    Returns:

    '''
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

def bootstrap_yan(benchmark,assets,realized_result):
    return bootstrap_ff(benchmark,assets,realized_result)

if __name__ == '__main__':
    benchmark,raw_factors=get_data()
    realized_result=pricing_assets(benchmark,raw_factors)
    # re=bootstrap_residual_independently(benchmark,raw_factors)
    # re1=bootstrap_residual_and_factor(benchmark,raw_factors)
    # re2=bootstrap_cross_sectional(benchmark,raw_factors)
    # re3=bootstrap_ff(benchmark,raw_factors)
    re4=bootstrap_kogan(benchmark,raw_factors,realized_result,100)






