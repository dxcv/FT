# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-03  14:13
# NAME:FT_hp-conditonal2.py
from functools import reduce

import pandas as pd
from config import DIR_TMP
from data.dataApi import get_filtered_ret
import os

from empirical.chordia.identify_anomalies1 import get_prominent_indicators
from empirical.config_ep import DIR_CHORDIA, DIR_DM, DIR_DM_INDICATOR, \
    DIR_DM_NORMALIZED, DIR_BASEDATA
from empirical.get_basedata import BENCHS
from tools import multi_process

# CONDITIONAL='ivol'


#conditional on idiosyncratical volatility
def get_indicator_direction(indname):
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)

    symbol=1 if at.at[indname,'capmM']>0 else -1
    return symbol

def get_comb_indicators(critic=2):
    iid='oepwqfaldagha'
    path=os.path.join(DIR_TMP,f'{iid}{critic}.pkl')
    # path=os.path.join(DIR_TMP,'dafj3e3ncsv.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        ind_names = get_prominent_indicators(critic)
        indicators = pd.concat([pd.read_pickle(
            os.path.join(DIR_DM_NORMALIZED,
                         ind + '.pkl')).stack() * get_indicator_direction(ind) for
                                ind in
                                ind_names], axis=1,
                               keys=ind_names)  # trick：revert the negative signal
        indicators.to_pickle(path)
        return indicators

def get_comb(cond_variable):
    indicators=get_comb_indicators()

    conditional=pd.read_pickle(os.path.join(DIR_BASEDATA,'normalized_conditional',cond_variable+'.pkl'))
    ret=get_filtered_ret().swaplevel()

    conditional=conditional.groupby('stkcd').shift(1)
    indicators=indicators.groupby('stkcd').shift(1)#trick: use the indicators of time t

    comb=pd.concat([indicators, ret, conditional], axis=1)
    comb=comb.dropna(subset=['ret_m',cond_variable])
    comb=comb.groupby('month_end').filter(lambda df:df.shape[0]>300)#trick: filter out months with too small sample
    comb=comb.fillna(0)
    print(cond_variable,len(comb.index.get_level_values('month_end').unique()))
    return comb


def _conditional_anomaly_return(args):
    comb,col,cond_variable=args
    #groupby conditional variable
    comb['gc']=comb.groupby('month_end',group_keys=False).apply(lambda df:pd.qcut(df[cond_variable],5,
                                                          labels=[f'g{i}' for i in range(1,6)]))
    # comb['gc']=comb.groupby('month_end',group_keys=False).apply(lambda df:pd.qcut(df[cond_variable].rank(method='first'),5,
    #                                                       labels=[f'g{i}' for i in range(1,6)]))

    #groupby factor
    comb['gf']=comb.groupby(['month_end','gc'],group_keys=False).apply(lambda df:pd.qcut(df[col].rank(method='first'),10,
                                labels=[f'g{i}' for i in range(1,11)]))

    stk=comb.groupby(['month_end', 'gc', 'gf']).apply(lambda df:df['ret_m'].mean()).unstack('gf')
    panel=(stk['g10'] - stk['g1']).unstack()
    panel.columns=panel.columns.astype(str)
    panel['all']=panel.mean(axis=1)
    panel['high-low']=panel['g5']-panel['g1']

    alpha=panel.mean()
    # t=panel.mean()/panel.std()
    t=panel.mean()/panel.sem() #trick: tvalue = mean / stderr,   stderr = std / sqrt(n-1) ,pd.Series.sem() = pd.Series.std()/pow(len(series),0.5)

    table=pd.concat([alpha,t],axis=1,keys=['alpha','t']).T
    # print(col)
    return table

def get_conditional_anomaly_return(cond_variable):

    comb = get_comb(cond_variable)


    indicators = [col for col in comb.columns if
                  col not in ['ret_m', cond_variable]]

    args_generator=((comb,col,cond_variable) for col in indicators)

    # tables=[]
    # for args in args_generator:
    #     tables.append(_conditional_anomaly_return(args))#fixme:
    tables=multi_process(_conditional_anomaly_return,args_generator,20)
    table5=pd.concat(tables,axis=0,keys=indicators)
    table5.to_csv(os.path.join(DIR_CHORDIA, f'table5_{cond_variable}.csv'))



def test_all_conditional_indicators():
    fns = os.listdir(os.path.join(DIR_BASEDATA, 'normalized_conditional'))
    conds = [fn[:-4] for fn in fns]
    for cond in conds:
        get_conditional_anomaly_return(cond)
        print(cond)

def analyze_turnover():
    fns=os.listdir(DIR_CHORDIA)
    fns=[fn for fn in fns if fn.startswith('table5_T')]
    dfs = [pd.read_csv(os.path.join(DIR_CHORDIA, fn), index_col=0) for fn in
           fns]
    df = pd.concat(dfs, keys=[fn[:-4] for fn in fns]) #fixme:
    df = df[df.iloc[:, 0] == 't']
    df['abs'] = df['high-low'].abs()
    df = df.sort_values('abs', ascending=False)
    df = df.iloc[:, 1:]
    df.index.names = ['cond_variable', 'indicator']
    lt2 = df.groupby('cond_variable').apply(
        lambda df: df[df['abs'] >= 2].shape[0]).sort_values(ascending=False)
    target = df.loc[(lt2.index[0], slice(None)), :]
    target.to_csv(os.path.join(DIR_TMP,'target1.csv'))

def analyze_ivol():
    fns=os.listdir(DIR_CHORDIA)
    fns=[fn for fn in fns if fn.startswith('table5_')]
    dfs=[pd.read_csv(os.path.join(DIR_CHORDIA,fn),index_col=0) for fn in fns]
    df=pd.concat(dfs,keys=[fn[:-4] for fn in fns])
    df=df[df.iloc[:,0]=='t']
    df['abs']=df['high-low'].abs()
    df=df.sort_values('abs',ascending=False)
    df=df.iloc[:,1:]
    df.index.names=['cond_variable','indicator']
    lt2=df.groupby('cond_variable').apply(lambda df:df[df['abs']>=2].shape[0]).sort_values(ascending=False)
    target=df.loc[(lt2.index[0],slice(None)),:]
    target.to_csv(os.path.join(DIR_TMP,'target.csv'))
    df.to_csv(os.path.join(DIR_CHORDIA,'table5_all.csv'))


def main():
    test_all_conditional_indicators()

if __name__ == '__main__':
    main()


