# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-11  19:32
# NAME:FT_hp-summary.py
import os
import pandas as pd
import pickle
from config import DIR_TMP
from empirical.config_ep import DIR_DM, DIR_CHORDIA
from empirical.get_basedata import BENCHS
from tools import multi_process


def save(df,name):
    df.to_pickle(os.path.join(DIR_CHORDIA,'writing',name+'.pkl'))
    df.to_csv(os.path.join(DIR_CHORDIA,'writing',name+'.csv'))



def _task_get_summary(name):
    s=pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',name+'.pkl'))['tb']
    return pd.Series([s.mean(),s.mean()/s.sem()],index=['alpha','t'],name=name)

def get_alpha_t():
    names=pickle.load(open(os.path.join(DIR_DM,'playing_indicators.pkl'),'rb'))
    alpha_t=pd.concat(multi_process(_task_get_summary,names,n=20),axis=1).T
    # alpha_t.to_pickle(os.path.join(DIR_TMP,'alpha_t.pkl'))

_get_fname=lambda x:x.split('-')[1]

def _stat_alpha(s):
    return pd.Series([int(s.shape[0]), s.mean(), s.median(),
                      s.std(), s.min(), s.max(),
                      len(s[s.abs() > 0.005]),
                      # len(s[s['alpha'].abs()>0.005])/s.shape[0],
                      len(s[s.abs() > 0.01])],
                     # len(s[s['alpha'].abs()>0.01])/s.shape[0]],
                     index=['N','Mean','Median','Std','Min','Max','|alpha|>0.5%','|alpha|>1.0%'])

def _stat_t(s):
    return pd.Series([int(s.shape[0]), s.mean(),
                      s.median(),
                      s.std(), s.min(),
                      s.max(),
                      len(s[s.abs() > 1.96]),
                      # len(s[s['t'].abs()>0.005])/s.shape[0],
                      len(s[s.abs() > 2.57])],
                     # len(s[s['t'].abs()>0.01])/s.shape[0]],
                     index=['N', 'Mean', 'Median', 'Std', 'Min', 'Max',
                            '|t|>1.96', '|t|>2.57'])

def get_raw_ret_sumary():
    alpha_t=get_alpha_t()
    # alpha_t=pd.read_pickle(os.path.join(DIR_TMP, 'alpha_t.pkl'))
    alpha_t.index.name= 'name'
    alpha_t=alpha_t.reset_index()

    alpha_t['category']=alpha_t['name'].map(_get_fname)

    panelA=alpha_t.groupby('category')['alpha'].apply(_stat_alpha).unstack()
    panelB=alpha_t.groupby('category')['t'].apply(_stat_t).unstack()

    save(panelA,'summary_panelA')
    save(panelB,'summary_panelB')

def get_abnormal_alpha_summary():
    names=pickle.load(open(os.path.join(DIR_DM,'playing_indicators.pkl'),'rb'))

    panels=[]
    for bench in BENCHS:
        at=pd.read_pickle(os.path.join(DIR_CHORDIA,f'at_{bench}.pkl'))
        at.index.name='name'
        at=at.loc[names]
        at=at.reset_index()
        at['category']=at['name'].map(_get_fname)
        panel=at.groupby('category')[bench].apply(_stat_t).unstack()
        panel.loc['all']=_stat_t(at[bench])
        panels.append(panel)

    table=pd.concat(panels,axis=0,keys=BENCHS)
    save(table,'summary_abnormal_alpha.csv')

def get_hist():
    for bench in BENCHS:
        at=pd.read_pickle(os.path.join(DIR_CHORDIA,f'at_{bench}.pkl'))

        at.hist(bins=100,density=True).get_figure().show()


def get_fmt_summary():
    names=pickle.load(open(os.path.join(DIR_DM,'playing_indicators.pkl'),'rb'))
    fmt=pd.read_pickle(os.path.join(DIR_CHORDIA,'fmt.pkl'))

    inter=[nm for nm in fmt.index]

    fmt['category']=list(map(_get_fname,fmt.index.tolist()))

    panels=[]
    for bench in BENCHS:
        panel=fmt.groupby('category')[bench].apply(_stat_t).unstack()
        panel.loc['all']=_stat_t(fmt[bench])
        panels.append(panel)
        print(bench)
    table=pd.concat(panels,axis=0,keys=BENCHS)
    save(table,'fmt_slope_summary')









def main():
    get_raw_ret_sumary()
    get_abnormal_alpha_summary()

# if __name__ == '__main__':
#     main()








