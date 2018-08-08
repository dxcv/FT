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
from tools import multi_task


def _task(fn):
    criterias = pd.read_csv(
        os.path.join(DIR_MIXED_SIGNAL_BACKTEST, fn, 'hedged_perf.csv'),
        index_col=0, header=None).iloc[:, 0]
    ret = pd.read_csv(
        os.path.join(DIR_MIXED_SIGNAL_BACKTEST, fn, 'hedged_returns.csv'),
        index_col=0, header=None).iloc[:, 0]
    cum_ret = (1 + ret).cumprod()[-1] - 1
    criterias.loc['cumulative return'] = cum_ret
    yearly = pd.read_csv(
        os.path.join(DIR_MIXED_SIGNAL_BACKTEST, fn, 'hedged_perf_yearly.csv'),
        index_col=0).iloc[:, 0]
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
    results=multi_task(_task,fns)
    ss=[r[0] for r in results]
    rets=[r[1] for r in results]
    summary=pd.concat(ss,axis=1,sort=True,keys=fns).T

    # review: 对于任意两个趋势有持续正收益的序列相关性都会很强，这种直接用收益率求相关性的方式不太准确

    # _ca=pd.concat([alpha]+rets,axis=1,keys=['alpha']+fns).dropna()
    # _c1=pd.concat([s1]+rets,axis=1,keys=['s1']+fns).dropna()
    # _c2=pd.concat([s2]+rets,axis=1,keys=['s2']+fns).dropna()

    # summary['corr_alpha']=np.np.corrcoef(_c1,rowvar=False)[:,0]
    # summary['corr_s1']=np.corrcoef(_c1,rowvar=False)[:,0]
    # summary['corr_s2']=np.corrcoef(_c2,rowvar=False)[:,0]

    summary['corr_alpha']=pd.concat([alpha]+rets,axis=1,keys=['alpha']+fns).dropna().corr()['alpha']
    summary['corr_s1']=pd.concat([s1]+rets,axis=1,keys=['s1']+fns).dropna().corr()['s1']
    summary['corr_s2']=pd.concat([s2]+rets,axis=1,keys=['s2']+fns).dropna().corr()['s2']
    summary=summary.sort_values('cumulative return',kind='mergesort',ascending=False)
    summary.to_csv(os.path.join(DIR_MIXED_SUM,'summary.csv'))





if __name__ == '__main__':
    summarize()

