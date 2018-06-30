# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-24  19:57
# NAME:FT-summarize_result.py
from config import SINGLE_D_CHECK, SINGLE_D_SUMMARY
import os
import pandas as pd
from data.dataApi import read_local
import numpy as np


def compare(fns,nlargest=None):
    '''

    Args:
        fns:fns to compare
        critera:
        nlargest:
        benchmark:

    Returns:

    '''
    b_dfs=[]
    g_des_dfs=[]
    beta_dfs=[]
    g_ret_dfs=[]
    for fn in fns:
        directory=os.path.join(SINGLE_D_CHECK,fn)
        beta_t_ic_des=pd.read_csv(os.path.join(directory,'beta_t_ic_des.csv'),header=None,index_col=0)
        beta_t_ic_des.columns=[fn]

        try:
            g_ret_des_tmb=pd.read_csv(os.path.join(directory,'g_ret_des.csv'),index_col=0).loc['g10_g1'].to_frame() #review:g rather than t
        except:
            g_ret_des_tmb = \
            pd.read_csv(os.path.join(directory, 't_ret_des.csv'),
                        index_col=0).loc[
                'g10_g1'].to_frame()  # review:g rather than t
        g_ret_des_tmb.columns=[fn]

        g_ret=pd.read_csv(os.path.join(directory,'g_ret.csv'),index_col=0,parse_dates=True)['g10_g1'].to_frame()
        g_ret.columns=[fn]

        beta=pd.read_csv(os.path.join(directory,'beta_t_ic.csv'),index_col=0,parse_dates=True)['beta'].to_frame()
        beta.columns=[fn]

        b_dfs.append(beta_t_ic_des)
        g_des_dfs.append(g_ret_des_tmb)
        beta_dfs.append(beta)
        g_ret_dfs.append(g_ret)

    g=pd.concat(g_ret_dfs,axis=1)
    b=pd.concat(beta_dfs,axis=1)

    beta_des=pd.concat(b_dfs,axis=1).T
    g_des=pd.concat(g_des_dfs,axis=1).T

    factors=beta_des['Return Mean'].abs().sort_values(ascending=False).index
    if nlargest:
        factors=factors[:nlargest]
        beta_des=beta_des.loc[factors]
        g_des=g_des.loc[factors]

    fig_layer=g[factors].cumsum().plot(figsize=(16,8)).get_figure()
    fig_beta=b[factors].cumsum().plot(figsize=(16,8)).get_figure()
    corr=b[factors].corr()
    return g_des,beta_des,corr,fig_layer,fig_beta

    # if abs_fig:
    #     sign=np.where(g_des[critera]>0,1,-1)
    #     g=g*sign
    #     factors=list(g_des[critera].abs().sort_values(ascending=False).index)
    # else:
    #     factors=list(g_des[critera].sort_values(ascending=False).index)
    #
    # indice_m=read_local('indice_m')
    # zz500=indice_m['zz500_ret_1m'].resample('M').last()
    # g=pd.concat([g,zz500],axis=1,join='inner')
    #
    # if benchmark:
    #     factors+=['zz500_ret_1m']

def compare_by_category(category=None,nlargest=None):
    fns = os.listdir(SINGLE_D_CHECK)
    if category:
        fns = [fn for fn in fns if fn.startswith(category)]
    return compare(fns,nlargest=nlargest)


def compare_different_growth_function():
    fns=os.listdir(SINGLE_D_CHECK)
    fns=[fn for fn in fns if fn.startswith('G')]

    #get function set and indicator set
    type_f=[]
    type_indicator=[]
    for fn in fns:
        f=fn.split("__")[0][2:]
        indicator=fn.split('__')[1]
        if f not in type_f:
            type_f.append(f)
        if indicator not in type_indicator:
            type_indicator.append(indicator)

    #compare with different function
    for ind in type_indicator:
        target = []
        for fn in fns:
            indicator = fn.split('__')[1]
            if indicator==ind:
                target.append(fn)
        g_des, beta_des, corr, fig_layer, fig_beta=compare(target)
        print(ind)

    #compare with different indicator
    for f in type_f:
        target = []
        for fn in fns:
            func = fn.split("__")[0][2:]
            if func==f:
                target.append(fn)
        g_des, beta_des, corr, fig_layer, fig_beta=compare(target)
        print(f)

def analyze_growth():
    fns=os.listdir(SINGLE_D_CHECK)
    fns=[fn for fn in fns if fn.startswith('G')]

    #get function set and indicator set
    type_f=[]
    type_indicator=[]
    for fn in fns:
        f=fn.split("__")[0][2:]
        indicator=fn.split('__')[1]
        if f not in type_f:
            type_f.append(f)
        if indicator not in type_indicator:
            type_indicator.append(indicator)

    df=pd.DataFrame()
    for fn in fns:
        f = fn.split("__")[0][2:]
        indicator = fn.split('__')[1]
        ann=pd.read_csv(os.path.join(SINGLE_D_CHECK,fn,'g_ret_des.csv'),index_col=0).loc['g10_g1','annual']
        df.loc[f,indicator]=ann

    return df


def color_negative_red(val):
    color='red' if val<0 else 'black'
    return 'color:%s'%color

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def summarize_result(category=None):
    if category:
        directory=os.path.join(SINGLE_D_SUMMARY,category)
    else:
        directory=os.path.join(SINGLE_D_SUMMARY,'all')
    if not os.path.exists(directory):
        os.makedirs(directory)

    g_des,beta_des,corr,fig_layer,fig_beta=compare_by_category(category=category)
    fig_layer.savefig(os.path.join(directory,'layer.png'))
    fig_beta.savefig(os.path.join(directory,'beta.png'))
    g_des.to_excel(os.path.join(directory,'g_des.xlsx'))
    beta_des.to_excel(os.path.join(directory,'beta_des.xlsx'))
    corr.to_excel(os.path.join(directory,'corr.xlsx'))

    # corr.style.format('{:.4}').applymap(color_negative_red).\
    #     to_excel(os.path.join(directory,'corr.xlsx'),engine='openpyxl')


# summarize_result('T')
# summarize_result('V')
# summarize_result('Q')
# summarize_result('G')
summarize_result('C')




