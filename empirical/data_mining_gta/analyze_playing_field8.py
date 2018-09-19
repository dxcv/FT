# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-18  23:09
# NAME:FT_hp-analyze_playing_field8.py



import os
import pandas as pd
from empirical.config_ep import DIR_DM_GTA
from empirical.get_basedata import BENCHS, get_benchmark


DIR_ANALYSE=os.path.join(DIR_DM_GTA,'analyse')

def get_prominent_indicators(mode=1,critic=2):


    at=pd.read_pickle(os.path.join(DIR_ANALYSE,'at.pkl'))
    fmt=pd.read_pickle(os.path.join(DIR_ANALYSE,'fmt.pkl'))

    tp=pd.concat([pd.read_pickle(os.path.join(DIR_ANALYSE,'bootstrap',f'{bench_name}_bp.pkl')) for bench_name in BENCHS],
                 axis=1,keys=BENCHS,sort=True)

    tp[tp>0.99].notnull().sum()
    tp[tp<0.01].notnull().sum()

    inds1=at[abs(at)>critic].dropna().index.tolist()
    inds2=fmt[abs(fmt)>critic].dropna().index.tolist()
    inds=[ind for ind in inds1 if ind in inds2]
    len(inds) #10


    prominent=None
    if mode==1:
        prominent=inds1
    elif mode==2:
        prominent=inds2
    elif mode==3:
        prominent=inds


    # _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM_GTA,'port_ret','eq',x+'.pkl'))['tb']
    #
    # df=pd.concat([_get_s(ind) for ind in inds2],axis=1,keys=inds2)
    # df.cumsum().plot().get_figure().show()
    #
    # cr=df.corr().stack().sort_values()

    return prominent


#=================method3: cluster=========================

#==================method4: PLS============================

#==================================================================================================

#TODO: we should check absolute tvalue, since these signals are generated randomly, being negative or positive does not make any sense.

'''
1. 有些指标样本太少
2. fm 中的指标

'''