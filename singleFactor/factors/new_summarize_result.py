# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-24  19:57
# NAME:FT-new_summarize_result.py
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

    fig_layer=g[factors].cumsum().plot().get_figure()
    fig_beta=b[factors].cumsum().plot().get_figure()
    return g_des,beta_des,fig_layer,fig_beta

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

def summrize_sample():
    beta_des,g_des,fig_cumprod,fig_cumsum=compare_by_category(nlargest=10)
    fig_cumsum.show()
    fig_cumprod.show()

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
        beta_des, g_des, fig_cumprod, fig_cumsum=compare(target)
        fig_cumsum.savefig(os.path.join(SINGLE_D_SUMMARY,'growth',ind+'.png'))
        print(ind)

    #compare with different indicator
    for f in type_f:
        target = []
        for fn in fns:
            func = fn.split("__")[0][2:]
            if func==f:
                target.append(fn)
        beta_des, g_des, fig_cumprod, fig_cumsum=compare(target)
        fig_cumsum.savefig(os.path.join(SINGLE_D_SUMMARY,'growth',f+'.png'))
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

# df=analyze_growth()

# _,_,_,fig_T=compare_by_category(category='T',nlargest=5)
# _,_,_,fig_Q=compare_by_category(category='Q',nlargest=5)
# _,_,_,fig_V=compare_by_category(category='V',nlargest=4)

# beta_des,g_des,fig_cumprod,fig_cumsum=compare_by_category(nlargest=20)


# fig_T.show()
# fig_Q.show()
# fig_V.show() #

# _,_,fig_T_sharpe,_=compare_by_category(category='T',nlargest=10,critera='sharpe')
# _,_,fig_T_annual,_=compare_by_category(category='T',nlargest=10,critera='annual')

# fig_T_sharpe.show()
# fig_T_annual.show()


_,_,fig_layer,fig_beta=compare_by_category(nlargest=20)
fig_layer.show()
fig_beta.show()
