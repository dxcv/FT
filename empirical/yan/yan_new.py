# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-23  20:06
# NAME:FT_hp-yan_new.py
from scipy import stats

from empirical.bootstrap import pricing_assets, bootstrap_yan
from empirical.config_ep import DIR_YAN
from empirical.get_basedata import get_data, BENCHS
from tools import multi_process
import pickle
import pandas as pd
import os


def simulate(bench,n=100):
    benchmark,assets=get_data(bench)
    realized_result=pricing_assets(benchmark,assets)
    rs=multi_process(bootstrap_yan,((benchmark,assets,realized_result) for i in range(n)),multi_paramters=True)
    result={}
    for ind in ['alpha','alpha_t','alpha_p']:
        df=pd.concat([r[ind] for r in rs],axis=1)
        # df.to_pickle(os.path.join(DIR_YAN,ind+'.pkl'))
        result[ind]=df
    pickle.dump(result,open(os.path.join(DIR_YAN,f'{bench}_{n}.pkl'),'wb'))

def get_bootstrap_pvalue(bench):
    simulate(bench)
    simulated=pickle.load(open(os.path.join(DIR_YAN,f'{bench}_100.pkl'),'rb'))
    benchmark, assets = get_data(bench)
    realized=pricing_assets(benchmark,assets)

    inds=['alpha','alpha_t','alpha_p']
    ss=[]
    for ind in inds:
        s=pd.Series([realized[ind].quantile(i/100) for i in range(1,101)],index=range(1,101))
        s.name=ind
        critical=pd.Series([simulated[ind].quantile(i/100).quantile(5/100) for i in range(1,101)],index=range(1,101))
        critical.name=f'critical0.05_{ind}'
        sp=pd.Series([stats.percentileofscore(simulated[ind].quantile(i/100),realized[ind].quantile(i/100))/100
                     for i in range(1,101)],index=range(1,101))
        sp.name=f'pvalue_{ind}'
        ss.append(s)
        ss.append(critical)
        ss.append(sp)

    df=pd.concat(ss,axis=1)
    df.to_csv(os.path.join(DIR_YAN,f'result_{bench}.csv'))

def run():
    for bench in BENCHS:
        if bench not in ['ffcM']:
            #fixme: for ffcM，the slope for "mom" can be nan due to the high collinearity with the factors in assets
            get_bootstrap_pvalue(bench)

if __name__ == '__main__':
    run()
