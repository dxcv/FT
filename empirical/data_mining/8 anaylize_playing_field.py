# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-15  19:54
# NAME:FT_hp-8 anaylize_playing_field.py

from empirical.config_ep import DIR_DM, DIR_CHORDIA, DIR_DM_NORMALIZED, \
    PERIOD_THRESH, DIR_BASEDATA, DIR_YAN
import os
import pandas as pd
from empirical.get_basedata import BENCHS, get_benchmark
from empirical.utils import align_index
from empirical.yan.yan_new import get_realized
from tools import multi_process
import numpy as np
import pickle


def get_prominent_indicators(critic=3):
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)
    fmt=pd.read_pickle(os.path.join(DIR_CHORDIA,'fmt.pkl'))

    inds1=at[abs(at)>critic].dropna().index.tolist()
    inds2=fmt[abs(fmt)>critic].dropna().index.tolist()
    inds=[ind for ind in inds1 if ind in inds2]
    # len(inds) #26

    # inds1=at[at>CRITIC].dropna().index.tolist()
    # inds2=at[at<-CRITIC].dropna().index.tolist()
    # inds3=fmt[fmt>CRITIC].dropna().index.tolist()
    # inds4=fmt[fmt<-CRITIC].dropna().index.tolist()

    # _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    # df=pd.concat([_get_s(ind) for ind in inds],axis=1,keys=inds)
    # df.cumsum().plot().get_figure().show()

    # cr=df.corr().stack().sort_values()
    return inds

#=================method 0: select manually=================================
def get_prominent_anomalies0():
    alpha_t = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)
    fmt = pd.read_pickle(os.path.join(DIR_CHORDIA, 'fmt.pkl'))

    CRITIC = 3

    inds1 = alpha_t[alpha_t > CRITIC].dropna().index.tolist()
    inds2 = alpha_t[alpha_t < -CRITIC].dropna().index.tolist()
    inds3 = fmt[fmt > CRITIC].index.tolist()
    inds4 = fmt[fmt < -CRITIC].index.tolist()

    indicators=inds1+inds2
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    # df.cumsum().plot().get_figure().show()

    cr=df.corr().stack().sort_values()

    test_indicators=abs(cr).sort_values().index[0]

    myfactors=df[list(test_indicators)]
    return myfactors



#=================method3: cluster=========================

#==================method4: PLS============================

#==================================================================================================

#TODO: we should check absolute tvalue, since these signals are generated randomly, being negative or positive does not make any sense.

'''
1. 有些指标样本太少
2. fm 中的指标

'''