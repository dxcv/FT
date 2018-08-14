# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-09  13:28
# NAME:FT_hp-panel_regression.py
import os
import pandas as pd
import statsmodels.api as sm
import random
import numpy as np
from empirical.config_ep import DIR_KOGAN
from scipy import stats

from empirical.replication import get_raw_factors
from tools import multi_task

DIR_TMP=r'G:\FT_Users\HTZhang\FT\tmp'

def get_assets_and_candidates():
    assets=get_raw_factors()
    rpM=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata','rpM.pkl'))
    candidates=pd.concat([assets,rpM],axis=1).dropna()
    assets=assets.reindex(candidates.index)
    return assets,candidates

def _get_summary_statistics(assets, candidates,method='mean'):
    '''
    get SI for each of the candidates
    Args:
        assets:DataFrame
        candidates:DataFrame,assets and candidates should share the same time index.

    Returns:Series

    '''
    #TODO: when the pre-selected factors is not None
    a_b = assets.mean()  # cross section of regression intercepts for the baseline model
    s_b=assets.apply(lambda s:s.std()/pow(len(s),0.5))# cross section of standard errors for regression intercepts under the baseline model
    si_list = []
    cd_name_list = []
    for cd_name in candidates.columns:
        cd = candidates[cd_name]
        alpha_list = []
        _ast_names = []
        for ast_name in assets.columns:
            if ast_name != cd_name:#trick: do not
                Y = assets[ast_name]
                X = sm.add_constant(cd)
                r = sm.OLS(Y, X).fit()
                alpha = r.params['const']
                alpha_list.append(alpha)
                _ast_names.append(ast_name)

        a_g = pd.Series(alpha_list, index=_ast_names)

        comb = pd.concat([a_b, a_g, s_b], axis=1, keys=['a_b', 'a_g', 's_b'],
                         sort=True)

        comb = comb.dropna()
        if method=='mean':
            mean1=(comb['a_g'].abs()/comb['s_b']).mean()
            mean2=(comb['a_b'].abs()/comb['s_b']).mean()
            si_mean=(mean1-mean2)/mean2
            si_list.append(si_mean)
        elif method=='median':
            med1=(comb['a_g'].abs()/comb['s_b']).median()
            med2=(comb['a_b'].abs()/comb['s_b']).median()
            si_med=(med1-med2)/med2
            si_list.append(si_med)
        cd_name_list.append(cd_name)
    summary_statistics = pd.Series(si_list, cd_name_list)
    return summary_statistics

def get_summary_statistics_real(method='mean'):
    # step1: statistics for the realized data
    assets,candidates=get_assets_and_candidates()
    sum_stat_real=_get_summary_statistics(assets, candidates,method)
    return sum_stat_real

def _get_summary_statistics_sampled(args):
    assets, candidates,method,i=args
    # sample the time index
    sampled_index = random.choices(assets.index, k=assets.shape[0])

    sampled_assets = assets.loc[sampled_index]  # sampled assets
    sampled_candidates = candidates.loc[sampled_index]  # sampled candidates
    sum_stat_sample=_get_summary_statistics(sampled_assets, sampled_candidates,method)
    print(i)
    return sum_stat_sample

def get_sampled_summary_statistics(method='mean',n=1000):
    assets,candidates=get_assets_and_candidates()
    # orthogonalized candidates
    orth_candidates=candidates-candidates.mean()# trick: different with realized data,we demean the candidates at this place
    args_generator=((assets,orth_candidates,method,i) for i in range(n))
    df=pd.concat(multi_task(_get_summary_statistics_sampled,args_generator),axis=1)
    df.to_pickle(os.path.join(DIR_TMP,'sampled_{}_{}.pkl'.format(method,n)))

def analyze_result():
    method='mean'
    n=1000
    sum_real=get_summary_statistics_real()
    sampled=pd.read_pickle(os.path.join(DIR_TMP,'sampled_{}_{}.pkl'.format(method,n)))

    q5=sampled.quantile(0.05,axis=1)
    q95=sampled.quantile(0.95,axis=1)

    comb=pd.concat([sum_real,q5,q95],axis=1,keys=['realized','q5','q95'])
    comb['significant']=np.nan
    comb['significant'][comb['realized']<comb['q5']]=-1
    comb['significant'][comb['realized']>comb['q95']]=1

    sampled.mean(axis=1)

    p_list=[]
    for ind in comb.index:
        x=comb.at[ind,'realized']
        p_list.append(stats.percentileofscore(sampled.loc[ind],x)/100)

    comb['pvalue']=p_list
    #TODO: how to explain those positively significant factors?




# preselected=['G_pct_4__s_fa_roe']




# if __name__ == '__main__':
#     get_sampled_summary_statistics(method='median')
#     get_sampled_summary_statistics(method='mean')



#TODO:set mkt as the first pre-selected factor

