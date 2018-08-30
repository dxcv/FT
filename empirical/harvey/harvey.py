# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-09  13:28
# NAME:FT_hp-harvey.py
import os
import random

import numpy as np
import pandas as pd
import statsmodels.api as sm
from empirical.config_ep import DIR_KOGAN
from empirical.kogan.kogan_part1 import get_raw_factors
from scipy import stats
from tools import multi_process

DIR_TMP=r'G:\FT_Users\HTZhang\empirical\harvey\tmp'


def _get_summary_statistics_old(assets, candidates,method='mean'):
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

def calculate_ab_and_sb(assets,pre_selected):
    if pre_selected is None:
        #TODO: when the pre-selected factors is not None
        a_b = assets.mean()  # cross section of regression intercepts for the baseline model
        s_b=assets.apply(lambda s:s.std()/pow(len(s),0.5))# cross section of standard errors for regression intercepts under the baseline model
    else:
        alpha_list=[]
        stderr_list=[]
        _ast_names=[]

        for ast_name in assets.columns:
            if ast_name not in pre_selected.columns:
                Y=assets[ast_name]
                X=sm.add_constant(pre_selected)
                r=sm.OLS(Y,X).fit()
                alpha_list.append(r.params['const'])
                stderr_list.append(r.bse['const'])
                _ast_names.append(ast_name)
        a_b=pd.Series(alpha_list,index=_ast_names)
        s_b=pd.Series(stderr_list,index=_ast_names)
    return a_b,s_b

def calcualte_SI(assets,pre_selected,candidates,method):
    a_b,s_b=calculate_ab_and_sb(assets,pre_selected)
    si_list=[]
    cd_name_list=[]
    for cd_name in candidates.columns:
        cd=candidates[cd_name].to_frame()
        if pre_selected is None:
            augmented=cd
        else:
            augmented=pd.concat([pre_selected,cd],axis=1)

        ast_names=[col for col in assets.columns if col not in augmented.columns]
        #matrix operation, run n OLS regression at the same time
        Y=assets[ast_names].values
        X=sm.add_constant(augmented).values
        _alpha=np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)[0]
        a_g=pd.Series(_alpha,index=ast_names)

        comb=pd.concat([a_b,a_g,s_b],axis=1,keys=['a_b','a_g','s_b'],sort=True)
        comb=comb.dropna()
        if method == 'mean':
            mean1 = (comb['a_g'].abs() / comb['s_b']).mean()
            mean2 = (comb['a_b'].abs() / comb['s_b']).mean()
            si_mean = (mean1 - mean2) / mean2
            si_list.append(si_mean)
        elif method == 'median':
            med1 = (comb['a_g'].abs() / comb['s_b']).median()
            med2 = (comb['a_b'].abs() / comb['s_b']).median()
            si_med = (med1 - med2) / med2
            si_list.append(si_med)
        cd_name_list.append(cd_name)

    summary_statistics_real = pd.Series(si_list, cd_name_list)
    return summary_statistics_real

def get_pseudo_factors(pre_selected,candidates):
    if pre_selected is None:
        pseudo_factors=candidates-candidates.mean()
    else:
        alpha_list = []
        for cd_name in candidates.columns:
            Y = candidates[cd_name]
            X = sm.add_constant(pre_selected)
            r = sm.OLS(Y, X).fit()
            alpha_list.append(r.params['const'])
        pseudo_factors = candidates - pd.Series(alpha_list,
                                                index=candidates.columns)
    return pseudo_factors

def get_sampled_data(assets,pre_selected,pseudo_factors):
    '''
    project the candidates on pre-selected factors and constuct the pseudo factors by substructing the
    regression intercept from the candidates
    '''
    #sample the time index
    sampled_index = random.choices(assets.index, k=assets.shape[0])
    sampled_assets=assets.loc[sampled_index]
    if pre_selected is None:
        sampled_pre_selected=None
    else:
        sampled_pre_selected = pre_selected.loc[sampled_index]
    sampled_ps_factors=pseudo_factors.loc[sampled_index]
    return sampled_assets,sampled_pre_selected,sampled_ps_factors

def _resample_onetime(args):
    assets, pre_selected, pseudo_factors, method,i=args
    sampled_assets, sampled_pre_selected, sampled_ps_factors = get_sampled_data(
        assets, pre_selected, pseudo_factors)
    si_sampled = calcualte_SI(sampled_assets, sampled_pre_selected,
                              sampled_ps_factors, method)
    print(i)
    return si_sampled

def resample_ntime(assets, pre_selected, candidates, method, n=1000):
    '''

    Args:
        assets:
        pre_selected:
        candidates:
        method:
        n:

    Returns:DataFrame, each column denotes a sampled result

    '''
    if pre_selected is None:
        path=os.path.join(DIR_TMP,'samples_{}_{}.pkl'.format(method,n))
    else:
        path=os.path.join(DIR_TMP,'samples_{}_{}_{}.pkl'.format(method,n,''.join(pre_selected.columns)))

    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        pseudo_factors = get_pseudo_factors(pre_selected, candidates)
        args_generator=((assets,pre_selected,pseudo_factors,method,i) for i in range(n))
        si_sampled=pd.concat(multi_process(_resample_onetime, args_generator, 32), axis=1)
        si_sampled.to_pickle(path)
        return si_sampled

def get_three_base(pre_names=None):
    raw_factors = get_raw_factors()
    rpM = pd.read_pickle(os.path.join(DIR_KOGAN, 'basedata', 'rpM.pkl'))
    comb = pd.concat([raw_factors, rpM], axis=1).dropna()

    assets = comb[raw_factors.columns]
    if pre_names:
        pre_selected = comb[pre_names]
        candidates = comb[[col for col in comb.columns if col not in pre_names]]
    else:
        pre_selected=None
        candidates=comb.copy()
    return assets,pre_selected,candidates

def get_pvalue(n,method,pre_names=None):
    assets, pre_selected, candidates=get_three_base(pre_names)

    si_real = calcualte_SI(assets, pre_selected, candidates,method)
    si_sampled_df=resample_ntime(assets, pre_selected, candidates, method, n)

    q5=si_sampled_df.quantile(0.05,axis=1)
    q95=si_sampled_df.quantile(0.95,axis=1)

    comb=pd.concat([si_real,q5,q95],axis=1,keys=['realized','q5','q95'])
    comb['significant']=np.nan
    comb['significant'][comb['realized']<comb['q5']]=-1
    comb['significant'][comb['realized']>comb['q95']]=1

    p_list=[]
    for ind in comb.index:
        x=comb.at[ind,'realized']
        p_list.append(stats.percentileofscore(si_sampled_df.loc[ind],x)/100)

    comb['pvalue']=p_list

    if pre_names is None:
        name='result_{}_{}.pkl'.format(n,method)
    else:
        name='result_{}_{}_{}.pkl'.format(n,method,''.join(pre_names))
    comb.to_pickle(os.path.join(DIR_TMP,name))
    #TODO: how to explain those positively significant factors?

def step1():
    n=1000
    method='mean'
    # pre_names = ['T__turnover1_relative_avg_60_30'] #or None
    pre_names=None
    get_pvalue(n,method,pre_names)

def analyse1():
    result1 = pd.read_pickle(os.path.join(DIR_TMP, r'result_1000_mean.pkl'))
    result1 = result1.sort_values('pvalue').sort_values('realized')
    sample1 = pd.read_pickle(os.path.join(DIR_TMP, 'samples_mean_1000.pkl'))
    sample1.min().quantile(0.05) #  multiple test
    stats.percentileofscore(sample1.min(), result1.iat[0,0]) / 100


def step2():
    n = 1000
    method = 'mean'
    pre_names = ['T__turnover1_relative_avg_60_30'] #or None
    get_pvalue(n, method, pre_names)


def analyse2():
    result2=pd.read_pickle(os.path.join(DIR_TMP,r'result_1000_mean_T__turnover1_relative_avg_60_30.pkl'))
    result2=result2.sort_values('pvalue').sort_values('realized')
    sample2=pd.read_pickle(os.path.join(DIR_TMP,'samples_mean_1000_T__turnover1_relative_avg_60_30.pkl'))
    sample2.min().quantile(0.05)
    stats.percentileofscore(sample2.min(), result2.iat[0,0]) / 100

def test_step3():
    n = 1000
    method = 'mean'
    pre_names = ['T__turnover1_relative_avg_60_30','C__est_oper_revenue_FTTM_to_est_baseshare_FTTM_chg_180']  # or None
    get_pvalue(n, method, pre_names)

def analyse3():
    result3=pd.read_pickle(os.path.join(DIR_TMP,r'result_1000_mean_T__turnover1_relative_avg_60_30C__est_oper_revenue_FTTM_to_est_baseshare_FTTM_chg_180.pkl'))
    result3=result3.sort_values('pvalue').sort_values('realized')
    sample3=pd.read_pickle(os.path.join(DIR_TMP,'samples_mean_1000_T__turnover1_relative_avg_60_30C__est_oper_revenue_FTTM_to_est_baseshare_FTTM_chg_180.pkl'))
    sample3.min().quantile(0.05)
    stats.percentileofscore(sample3.min(), result3.iat[0,0]) / 100

def debug():
    pre_names=None
    method='mean'
    assets, pre_selected, candidates=get_three_base(pre_names)

    si_real = calcualte_SI(assets, pre_selected, candidates,method)

# if __name__ == '__main__':
#     debug()


#TODO:

if __name__ == '__main__':
    # step1()
    # step2()
    test_step3()





# if __name__ == '__main__':
#     get_sampled_summary_statistics(method='median')
#     get_sampled_summary_statistics(method='mean')



#TODO:set mkt as the first pre-selected factor
#TODO:
#TODO: the left hand sid is excess return of the asset (refer to (6) in page 15)


