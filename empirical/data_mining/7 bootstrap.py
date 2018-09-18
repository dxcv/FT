# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-15  19:52
# NAME:FT_hp-7 bootstrap.py

import pandas as pd
from empirical.data_mining.dm_api import get_data
from empirical.get_basedata import BENCHS
from scipy import stats

from empirical.bootstrap import pricing_assets, bootstrap_yan
from empirical.config_ep import DIR_YAN
from tools import multi_process
import pickle
import os

def get_realized(bench_name):
    path=os.path.join(DIR_YAN, f'realized_{bench_name}.pkl')
    if os.path.exists(path):
        realized=pickle.load(open(path,'rb'))
    else:
        benchmark, assets = get_data(bench_name)
        realized=pricing_assets(benchmark,assets)
        pickle.dump(realized,open(path,'wb'))
    return realized


def simulate(bench_name, n):
    path=os.path.join(DIR_YAN, f'{bench_name}_{n}.pkl')
    if os.path.exists(path):
        return

    benchmark,assets=get_data(bench_name)
    # realized_result=pricing_assets(benchmark,assets)
    realized=get_realized(bench_name)

    # rs=[]
    # for i in range(10):
    #     r=bootstrap_yan(benchmark,assets,realized)
    #     rs.append(r)

    rs=multi_process(bootstrap_yan, ((benchmark,assets,realized) for i in range(n)),15,
                     multi_parameters=True) #review:

    result={}
    for ind in ['alpha','alpha_t','alpha_p']:
        df=pd.concat([r[ind] for r in rs],axis=1)
        # df.to_pickle(os.path.join(DIR_YAN,ind+'.pkl'))
        result[ind]=df
    pickle.dump(result, open(path, 'wb'))


def get_bootstrap_pvalue(bench_name, n=1000):
    simulate(bench_name, n)
    simulated=pickle.load(open(os.path.join(DIR_YAN, f'{bench_name}_{n}.pkl'), 'rb'))
    realized=get_realized(bench_name)

    inds=['alpha','alpha_t','alpha_p']
    ss=[]
    for ind in inds:
        s=pd.Series([realized[ind].quantile(i/100) for i in range(101)],index=range(101))
        s.name=ind

        critical1=pd.Series([simulated[ind].quantile(i/100).quantile(5/100) for i in range(101)],index=range(101))
        critical1.name=f'critical0.05_{ind}'

        critical2=pd.Series([simulated[ind].quantile(i/100).quantile(95/100) for i in range(101)],index=range(101))
        critical2.name=f'critical0.95_{ind}'


        sp=pd.Series([stats.percentileofscore(simulated[ind].quantile(i/100),realized[ind].quantile(i/100))/100
                     for i in range(101)],index=range(101))
        sp.name=f'pvalue_{ind}'

        ss.append(s)
        ss.append(critical1)
        ss.append(critical2)
        ss.append(sp)

    '''The result is really different between ff3M and hxz4M'''
    df=pd.concat(ss,axis=1)
    df.to_csv(os.path.join(DIR_YAN, f'result_{bench_name}.csv'))

def bootstrap_t_pvalue(bench_name):
    # benchmark, assets = get_data(bench_name)
    simulated=pickle.load(open(os.path.join(DIR_YAN, f'{bench_name}_1000.pkl'), 'rb'))

    b_at=simulated['alpha_t']

    realized=get_realized(bench_name)

    r_at=realized['alpha_t']

    b_p=pd.Series([stats.percentileofscore(b_at.loc[ind],r_at[ind])/100.0 for ind in r_at.index],index=r_at.index)
    b_p.to_pickle(os.path.join(DIR_YAN,f'{bench_name}_bp.pkl'))

    # c5=b_at.quantile(0.05,axis=1)
    # c95=b_at.quantile(0.95,axis=1)
    # target=r_at[(r_at<c5) | (c95<r_at)]
    #
    # target=target[abs(target)>3]
    # len(target)

    # indicators=get_prominent_indicators()
    # len(indicators)
    # len([ind for ind in indicators if ind in target.index])

# for bench_name in BENCHS:
#     bootstrap_t_pvalue(bench_name)
#     print(bench_name)

def debug():
    bench_name='hxz4M'
    get_bootstrap_pvalue(bench_name,1000)


def main():
    for bench_name in BENCHS:
        # get_bootstrap_pvalue(bench_name)
        bootstrap_t_pvalue(bench_name)

if __name__ == '__main__':
    # main()
    debug()



