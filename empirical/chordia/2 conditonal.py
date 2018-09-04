# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-03  14:13
# NAME:FT_hp-2 conditonal.py
import pandas as pd
from config import DIR_TMP
from data.dataApi import get_filtered_ret
import os

from empirical.config_ep import DIR_CHORDIA, DIR_DM, DIR_DM_INDICATOR, \
    DIR_DM_NORMALIZED
from empirical.get_basedata import BENCHS

CONDITIONAL='ivol'

def get_anomaly_indicators():
    path=os.path.join(DIR_TMP,'opeqtda.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
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

        def _get_s(x):
            s=pd.read_pickle(os.path.join(DIR_DM_NORMALIZED,f'{x}.pkl')).stack()
            s.name=x
            return s

        # _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM_NORMALIZED,f'{x}.pkl'))
        ss=[]
        for ind in indicators:
            s=_get_s(ind)
            ss.append(_get_s(ind))
            print(s.index.names,s.shape)
        df=pd.concat(ss,axis=1)#review: this function is really slowly
        df.to_pickle(path)
        return df

#conditional on idiosyncratical volatility

def get_comb():
    path=os.path.join(DIR_TMP,'dafdakfec11.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        anomalies=get_anomaly_indicators()

        conditional = pd.read_pickle(r'G:\FT_Users\HTZhang\haitong\standardized\{}.pkl'.format(CONDITIONAL)).stack()
        conditional.name= 'conditional'

        ret=get_filtered_ret().swaplevel()

        anomalies=anomalies.groupby('stkcd').shift(1)#trick: use the indicators of time t
        comb=pd.concat([anomalies, ret, conditional], axis=1)
        comb=comb.dropna(subset=['ret_m','conditional'])
        comb=comb.fillna(0)

        comb.to_pickle(path)
        return comb



def get_conditional_anomaly_return():
    comb = get_comb()

    indicators = [col for col in comb.columns if
                  col not in ['ret_m', 'conditional', 'g']]

    tables=[]
    for col in indicators:
        comb['g0']=comb.groupby('month_end',group_keys=False).apply(lambda df:pd.qcut(df['conditional'],5,
                                                          labels=[f'g{i}' for i in range(1,6)]))
        comb['g1']=comb.groupby(['month_end','g0'],group_keys=False).apply(lambda df:pd.qcut(df[col].rank(method='first'),10,
                                    labels=[f'g{i}' for i in range(1,11)]))

        stk=comb.groupby(['month_end', 'g0', 'g1']).apply(lambda df:df['ret_m'].mean()).unstack('g1')
        panel=(stk['g10'] - stk['g1']).unstack()
        panel.columns=panel.columns.astype(str)
        panel['all']=panel.mean(axis=1)
        panel['high-low']=panel['g5']-panel['g1']


        alpha=panel.mean()
        t=panel.mean()/panel.std()
        table=pd.concat([alpha,t],axis=1,keys=['alpha','t']).T
        tables.append(table)
        print(col)

    table5=pd.concat(tables,axis=0,keys=indicators)
    table5.to_csv(os.path.join(DIR_CHORDIA,'table5.csv'))

if __name__ == '__main__':
    get_conditional_anomaly_return()

