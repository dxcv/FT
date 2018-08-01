# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-31  20:55
# NAME:FT_hp-pricing_performance.py
from empirical.config import DIR_KOGAN_RESULT, CRITICAL, NUM_FACTOR
import pandas as pd
import os

'''
p>CRITICAL  denotes the model can not pricing the assets perfectly
'''

def get_performance(weight=0):
    '''
    Args:
        weight:{0,1},0 denote equal weighed,1 denotes Characteristic Matching Frequency
    Returns:

    '''
    grs_model=pd.read_csv(os.path.join(DIR_KOGAN_RESULT,'grs_factor.csv'),index_col=0)

    grs_model[grs_model >= CRITICAL] = 1
    grs_model[grs_model < CRITICAL] = 0
    if weight==0:
        #performace1: equal weight
        perf=grs_model.sum(axis=1)/(grs_model.shape[1]-NUM_FACTOR+1)
    else:
        weight=1-grs_model.sum()/grs_model.shape[0]
        # weight=1-grs_model[grs_model>CRITICAL].notnull().sum()/grs_model.shape[0]
        perf=(grs_model*weight).sum(axis=1)/(grs_model.shape[1]-NUM_FACTOR+1)
    perf=perf.sort_values(kind='mergesort')
    return perf


