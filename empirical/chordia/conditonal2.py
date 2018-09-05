# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-03  14:13
# NAME:FT_hp-conditonal2.py
import pandas as pd
from config import DIR_TMP
from data.dataApi import get_filtered_ret
import os

from empirical.chordia.identify_anomalies1 import get_prominent_indicators
from empirical.config_ep import DIR_CHORDIA, DIR_DM, DIR_DM_INDICATOR, \
    DIR_DM_NORMALIZED
from empirical.get_basedata import BENCHS
from tools import multi_process

CONDITIONAL='ivol'

#conditional on idiosyncratical volatility
def get_indicator_direction(indname):
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)

    symbol=1 if at.at[indname,'capmM']>0 else -1
    return symbol

def get_comb():
    ind_names = get_prominent_indicators(critic=2)
    indicators = pd.concat([pd.read_pickle(
        os.path.join(DIR_DM_NORMALIZED, ind + '.pkl')).stack()*get_indicator_direction(ind) for ind in
                    ind_names], axis=1, keys=ind_names)

    conditional = pd.read_pickle(r'G:\FT_Users\HTZhang\haitong\standardized\{}.pkl'.format(CONDITIONAL)).stack()
    conditional.name= 'conditional'

    ret=get_filtered_ret().swaplevel()

    indicators=indicators.groupby('stkcd').shift(1)#trick: use the indicators of time t
    comb=pd.concat([indicators, ret, conditional], axis=1)
    comb=comb.dropna(subset=['ret_m','conditional'])
    comb=comb.fillna(0)
    return comb

def _conditional_anomaly_return(args):
    comb,col=args
    comb['gc']=comb.groupby('month_end',group_keys=False).apply(lambda df:pd.qcut(df['conditional'],5,
                                                          labels=[f'g{i}' for i in range(1,6)]))
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
    print(col)
    return table

def get_conditional_anomaly_return():
    comb = get_comb()

    indicators = [col for col in comb.columns if
                  col not in ['ret_m', 'conditional']]

    args_generator=((comb,col) for col in indicators)
    tables=multi_process(_conditional_anomaly_return,args_generator,20)
    table5=pd.concat(tables,axis=0,keys=indicators)
    table5.to_csv(os.path.join(DIR_CHORDIA,'table5.csv'))

def run():
    get_conditional_anomaly_return()

if __name__ == '__main__':
    run()



