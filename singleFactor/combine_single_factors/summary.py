# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-02  11:03
# NAME:FT_hp-summary.py
import os
import pandas as pd
import numpy as np

from config import DIR_MIXED_SIGNAL_BACKTEST, DIR_MIXED_SUM
from tools import multi_process


def _task(fn):
    criterias = pd.read_csv(
        os.path.join(DIR_MIXED_SIGNAL_BACKTEST, fn, 'hedged_perf.csv'),
        index_col=0, header=None).iloc[:, 0]
    ret = pd.read_csv(
        os.path.join(DIR_MIXED_SIGNAL_BACKTEST, fn, 'hedged_returns.csv'),
        index_col=0, header=None).iloc[:, 0]

    # cum_ret = (1 + ret).cumprod()[-1] - 1
    # criterias.loc['cumulative return'] = cum_ret
    try:
        yearly = pd.read_csv(
            os.path.join(DIR_MIXED_SIGNAL_BACKTEST, fn, 'hedged_perf_yearly.csv'),
            index_col=0,encoding='utf8').iloc[:, 0]
    except: #trick: some file are encoded as utf8 and the others are encoded as gbk
        yearly = pd.read_csv(
            os.path.join(DIR_MIXED_SIGNAL_BACKTEST, fn,
                         'hedged_perf_yearly.csv'),
            index_col=0,encoding='gbk').iloc[:, 0]
    s = pd.concat([criterias, yearly])
    print(fn)
    return s,ret

def summarize():
    alpha = pd.read_csv(os.path.join(DIR_MIXED_SUM, 'alpha.csv'), index_col=0,
                        parse_dates=True).iloc[:, 0]

    s1 = pd.read_csv(os.path.join(DIR_MIXED_SUM, 's1.csv'), index_col=0,
                        parse_dates=True).iloc[:, 0]

    s2 = pd.read_csv(os.path.join(DIR_MIXED_SUM, 's2.csv'), index_col=0,
                        parse_dates=True).iloc[:, 0]


    fns=os.listdir(DIR_MIXED_SIGNAL_BACKTEST)
    print(len(fns))
    fns=[fn for fn in fns if os.path.exists(os.path.join(DIR_MIXED_SIGNAL_BACKTEST,fn,'hedged_perf.csv'))]
    print(len(fns))

    results=multi_process(_task, fns)

    ss=[r[0] for r in results]
    rets=[r[1] for r in results]
    summary=pd.concat(ss,axis=1,sort=True,keys=fns).T

    # review: 对于任意两个趋势有持续正收益的序列相关性都会很强，这种直接用收益率求相关性的方式不太准确

    # _ca=pd.concat([alpha]+rets,axis=1,keys=['alpha']+fns).dropna()
    # _c1=pd.concat([s1]+rets,axis=1,keys=['s1']+fns).dropna()
    # _c2=pd.concat([s2]+rets,axis=1,keys=['s2']+fns).dropna()
    # summary['corr_alpha']=_ca.corrwith(_ca['alpha'])
    # summary['corr_s1']=_ca.corrwith(_c1['s1'])
    # summary['corr_s2']=_ca.corrwith(_c2['s2'])
    # summary=summary.sort_values('portfolio_total_return',kind='mergesort',ascending=False)
    # summary.to_csv(os.path.join(DIR_MIXED_SUM,'summary.csv'))

    comb=pd.concat([alpha,s1,s2]+rets,axis=1,keys=['alpha','s1','s2']+fns).dropna()
    # corr=comb.corr()
    # corr.to_csv(os.path.join(DIR_MIXED_SUM,'corr.csv'))
    comb.to_pickle(os.path.join(DIR_MIXED_SUM,'comb.pkl'))

# summary=pd.read_csv(os.path.join(DIR_MIXED_SUM,'summary.csv'),index_col=0)
# corr=pd.read_csv(os.path.join(DIR_MIXED_SUM,'corr.csv'),index_col=0)
#
# stk=corr.stack().sort_values()
#
# pct_to_float=lambda x:float(x[:-1])
#
# summary['2017_2018']=summary['2017'].map(pct_to_float)+summary['2018'].map(pct_to_float)
# inds1=summary['2017_2018'].nlargest(2000).index
# inds2=summary['sharp_ratio'].nlargest(2000).index
# inds3=summary[summary['portfolio_total_return']>=5.0].index
#
# inds=inds1.intersection(inds2).intersection(inds3)
#
# selected=corr.loc[inds,inds]
# stk=selected.stack().sort_values()
#
# stk.argmin()
#
#
# comb=pd.read_pickle(os.path.join(DIR_MIXED_SUM,'comb.pkl'))
# (1+comb[inds]).cumprod()


# stk=selected.stack().sort_values()




#
# if __name__ == '__main__':
#     summarize()

