# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  10:05
# NAME:FT_hp-get_basedata.py
import pandas as pd
import os
from empirical.config_ep import DIR_KOGAN

# rpM.pkl is copied from D:\zht\database\quantDb\researchTopics\assetPricing2_new\data\pkl_unfiltered
from tools import multi_process,multi_thread

rpM=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata','rpM.pkl'))

#benchmark models are all copied from D:\zht\database\quantDb\researchTopics\assetPricing2_new\data\pkl_unfiltered

BENCHS=['capmM','ff3M','ff5M','ff6M','ffcM','hxz4M']

_get_tb=lambda path:pd.read_pickle(path)['tb']

def get_raw_factors():
    '''get high-minu-low factors'''
    directory = os.path.join(DIR_KOGAN, 'port_ret', 'eq')
    fns = os.listdir(directory)
    arg_generator=(os.path.join(directory,fn) for fn in fns)
    # ss=multi_process(_get_tb, arg_generator)
    # ss=multi_thread(_get_tb,arg_generator)
    ss=[_get_tb(arg) for arg in arg_generator]
    raw_factors = pd.concat(ss, axis=1, keys=[fn[:-4] for fn in fns])
    #trick: delete those months with too small sample
    raw_factors = raw_factors.dropna(axis=0,thresh=int(raw_factors.shape[1] * 0.8))
    #trick: delete those factors with too short history
    raw_factors=raw_factors.dropna(axis=1,thresh=int(raw_factors.shape[0]*0.8))
    raw_factors = raw_factors.fillna(0)#trick:
    return raw_factors

def get_benchmark(name):
    df=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata',name+'.pkl'))
    return df

def get_data(bench='ff3M'):
    benchmark = get_benchmark(bench)
    raw_factors = get_raw_factors()
    base_index=benchmark.index.intersection(raw_factors.index)
    #trick: unify the index
    return benchmark.reindex(base_index),raw_factors.reindex(base_index)

