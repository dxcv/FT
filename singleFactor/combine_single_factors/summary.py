# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-02  11:03
# NAME:FT_hp-summary.py
import os
import pandas as pd

from config import DIR_MIXED_SIGNAL_BACKTEST, DIR_MIXED_SUM

def summarize():
    alpha = pd.read_csv(os.path.join(DIR_MIXED_SUM, 'alpha.csv'), index_col=0,
                        parse_dates=True).iloc[:, 0]

    s1 = pd.read_csv(os.path.join(DIR_MIXED_SUM, 's1.csv'), index_col=0,
                        parse_dates=True).iloc[:, 0]

    s2 = pd.read_csv(os.path.join(DIR_MIXED_SUM, 's2.csv'), index_col=0,
                        parse_dates=True).iloc[:, 0]


    fns=os.listdir(DIR_MIXED_SIGNAL_BACKTEST)
    ss=[]
    rets=[]
    for fn in fns:
        criterias=pd.read_csv(os.path.join(DIR_MIXED_SIGNAL_BACKTEST,fn,'hedged_perf.csv'),index_col=0,header=None).iloc[:,0]
        ret=pd.read_csv(os.path.join(DIR_MIXED_SIGNAL_BACKTEST,fn,'hedged_returns.csv'),index_col=0,header=None).iloc[:,0]
        cum_ret=(1+ret).cumprod()[-1]-1
        criterias.loc['cumulative return']=cum_ret
        yearly=pd.read_csv(os.path.join(DIR_MIXED_SIGNAL_BACKTEST,fn,'hedged_perf_yearly.csv'),index_col=0).iloc[:,0]
        s=pd.concat([criterias,yearly])
        ss.append(s)
        rets.append(ret)
    summary=pd.concat(ss,axis=1,sort=True,keys=fns).T

    summary['corr_alpha']=pd.concat([alpha]+rets,axis=1,keys=['alpha']+fns).dropna().corr()['alpha']
    summary['corr_s1']=pd.concat([s1]+rets,axis=1,keys=['s1']+fns).dropna().corr()['s1']
    summary['corr_s2']=pd.concat([s1]+rets,axis=1,keys=['s2']+fns).dropna().corr()['s2']
    summary=summary.sort_values('cumulative return',kind='mergesort',ascending=False)
    summary.to_csv(os.path.join(DIR_MIXED_SUM,'summary.csv'))

names=['alpha','s1','s2']
ss=[]
for name in names:
    s=pd.read_csv(os.path.join(DIR_MIXED_SUM,'{}.csv'.format(name)),index_col=0,parse_dates=True)
    if name=='alpha':
        s=s/100
    (1+s).cumprod().plot().get_figure().show()

    ss.append(s)

comb=pd.concat(ss,axis=1,keys=names)
(1+comb).cumprod().plot().get_figure().show()




# if __name__ == '__main__':
#     summarize()

