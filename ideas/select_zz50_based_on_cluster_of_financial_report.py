# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-14  18:58
# NAME:FT_hp-select_zz50_based_on_cluster_of_financial_report.py
import pandas as pd
import os
import numpy as np


from config import DIR_DM_TMP, DIR_TMP
from data.dataApi import read_local

def detect():
    financial = pd.read_pickle(os.path.join(DIR_DM_TMP, 'data.pkl'))
    fdmt = read_local('equity_fundamental_info').reset_index()
    ann=pd.pivot_table(financial,values='report_period',index='trd_dt',columns='stkcd',aggfunc='last')
    trading=read_local('equity_selected_trading_data').reset_index()
    indice=read_local('equity_selected_indice_ir')
    comb=pd.merge(trading,fdmt,how='left',on=['stkcd','trd_dt'])

    comb['downlimit']=np.where(comb['pctchange']<-9.5,True,False)
    comb['uplimit']=np.where(comb['pctchange']>=9.5,True,False)
    comb['islisted']=np.where(comb['trd_dt']>=comb['listdate'],True,False) #DEBUG: this way is not so rigorious,since comb does not cantain all the stocks listed in that time

    indicators=['cap','downlimit','uplimit','amount','islisted']
    ss=[]
    for indicator in indicators:
        panel=pd.pivot_table(comb,values=indicator,index='trd_dt',columns='stkcd')
        ss.append(panel.sum(axis=1))

    market_states=pd.concat(ss,axis=1,keys=indicators)


    #TODO: the amount of the index future
    market_states['n_ann']=ann.notnull().sum(axis=1)
    market_states['n_ann']=market_states['n_ann'].fillna(0)
    market_states['zz500']=indice['zz500'].pct_change()

    #adjust
    market_states['n_ann']/=market_states['islisted'].shift(1)
    market_states['amount']/=market_states['cap'].shift(1)
    market_states['downlimit']/=market_states['islisted'].shift(1)
    market_states['uplimit']/=market_states['islisted'].shift(1)


    target=market_states[['zz500','n_ann','amount','downlimit','uplimit']]
    corr=target.corr()
    target['n_ann'].plot().get_figure().show()
    target['n_ann'].nlargest(30)



financial = pd.read_pickle(os.path.join(DIR_DM_TMP, 'data.pkl'))

panel=financial.groupby(['report_period','trd_dt']).size().unstack('report_period')
for col in panel.columns:#DEBUG: some data is obviously wrong, since their announcement is smaller than report_period
    panel[col]=panel[col].where(panel.index>col,0)

panel=panel.fillna(0).cumsum()



fdmt = read_local('equity_fundamental_info').reset_index()
fdmt['islisted'] = np.where(fdmt['trd_dt'] >= fdmt['listdate'], True,
                            False)  # DEBUG: this way is not so rigorious,since comb does not cantain all the stocks listed in that time
islisted=pd.pivot_table(fdmt,values='islisted',index='trd_dt',columns='stkcd')
number=islisted.sum(axis=1)




panel=panel.reindex(number.index)


panel=panel['2008':]
panel=panel[[col for col in panel.columns if col.year>=2007]]


ratio=panel.divide(number,axis=0)
ratio=ratio.ffill()

ratio=ratio.apply(lambda s:s.where(s.index<=s.idxmax(),np.nan))


start,end=0.1,0.9
for start in [0.1,0.3,0.5]:
    for end in [0.5,0.6,0.7,0.8,0.9]:
        signal=ratio.copy()
        signal=signal.where(signal>=start,np.nan)
        signal=signal.where(signal<=end,np.nan)


        s=signal.notnull().sum(axis=1)
        s=s.where(s<=1,1)
        s=s.replace(0,-1)
        s=-s

        zz500 = read_local('equity_selected_indice_ir')['zz500_ret_d']

        comb=pd.concat([zz500,s],axis=1,keys=['zz500','signal'])
        comb['strategy']=comb['zz500']*comb['signal']

        cp=(1+comb[['zz500','strategy']]).cumprod()
        cp=np.log(cp)

        cp['relative']=cp['strategy']-cp['zz500']

        fig=cp.plot().get_figure()
        fig.savefig(os.path.join(DIR_TMP,'{}_{}.png'.format(start,end)))
        print(start,end)




import matplotlib.pyplot as plt


#TODO: think about the overlapped periods







'''
there are some data showing that the announcement date is smaller than report_period. It is abvious wrong!!!



'''





#TODO: capture the peak of n_ann





#TODO: divide the sample into two category based on size (or other characteristics) before testing

#TODO: the number of analyst expectation
#TODO: the number of announcement,
#TODO: the number of report

#TODO: other market states,such as market volatility and market sentiment
