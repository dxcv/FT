# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-05  11:34
# NAME:FT_hp-pca5.py
import pandas as pd
import os

from empirical.chordia.identify_anomalies1 import pricing_all_factors, \
    get_prominent_indicators
from empirical.config_ep import DIR_CHORDIA, DIR_DM
from empirical.get_basedata import BENCHS, get_benchmark
from empirical.utils import get_pca_factors


def get_at_pca():
    indicators = get_prominent_indicators()
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']
    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    pca=get_pca_factors(df,2)
    ff3=get_benchmark('ff3M')
    pca_model = pd.concat([ff3, pca], axis=1).dropna()
    results = pricing_all_factors(pca_model)
    results.to_pickle(os.path.join(DIR_CHORDIA, 'at_pca.pkl'))

    (abs(results>2)).sum()
    (abs(results>3)).sum()
