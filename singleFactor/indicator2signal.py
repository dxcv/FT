# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-16  09:37
# NAME:FT_hp-indicator2signal.py
from config import SINGLE_D_INDICATOR, DIR_SIGNAL, LEAST_CROSS_SAMPLE, DIR_ROOT
import os
import pandas as pd
import numpy as np
from data.dataApi import read_local
from singleFactor.singleTools import convert_indicator_to_signal
from tools import multi_process, outlier, z_score, neutralize
START='2006'

def indicator2signal(name):
    try:
        df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
        signal = convert_indicator_to_signal(df, name)
        signal.to_pickle(os.path.join(DIR_SIGNAL, name + '.pkl'))
        print(name)
    except:
        print('wrong!----------->{}'.format(name))

def convert_whole_history():
    fns=os.listdir(SINGLE_D_INDICATOR)
    names=[fn[:-4] for fn in fns]
    print('total number: {}'.format(len(names)))
    checked=[fn[:-4] for fn in os.listdir(DIR_SIGNAL)]
    names=[n for n in names if n not in checked]
    print('unchecked: {}'.format(len(names)))
    multi_process(indicator2signal,names,10)
    # for name in names:
    #     indicator2signal(name)

    # pool=multiprocessing.Pool(1)
    # pool.map(indicator2signal,names)

def adjust_fdmt():
    fdmt = read_local('equity_fundamental_info')[['type_st','young_1year','wind_indcd','cap']]
    fdmt=fdmt[(~fdmt['type_st']) & (~fdmt['young_1year'])] # 剔除st 和上市不满一年的数据
    fdmt=fdmt.dropna(subset=['wind_indcd'])
    fdmt['wind_2']=fdmt['wind_indcd'].apply(str).str.slice(0,6)
    fdmt['ln_cap']=np.log(fdmt['cap'])
    return fdmt

fdmt=adjust_fdmt()

def ind2sig_once(indicator_s):
    '''

    Args:
        indicator_s:Series, with the index.name as date

    Returns:

    '''
    date=indicator_s.name

    # name='anyname'
    # ind=indicator_s.loc[date]
    # ind.name=name

    ft= fdmt.loc[(slice(None), date), :]
    ft.index=ft.index.droplevel(level=1)
    comb=pd.concat([ft, indicator_s], axis=1, join='inner').dropna()
    if comb.shape[0]>=LEAST_CROSS_SAMPLE:
        comb[date] = outlier(comb[date])
        comb[date] = z_score(comb[date])
        comb = comb.join(pd.get_dummies(comb['wind_2'], drop_first=True))
        industry = list(np.sort(comb['wind_2'].unique()))[1:]
        signal = neutralize(comb, date,
                            industry)  # trick: signal calcuted use the information of time t, we can only trade on time t+1 based on this signal
        print(date)
        return signal
    else:
        print(f'{date}--->too small sample')
        # return pd.Series(name=date)

def ind2sig_period(indicator_df, start=None, end=None):
    if start is None:
        start=indicator_df.index[0]
    if end is None:
        end=indicator_df.index[-1]
    indicator_df= indicator_df[start:end]
    dates=indicator_df.index.tolist()

    # ss=[]
    # for date in dates:
    #     ind_s=indicator_df.loc[date]
    #     ss.append(ind2sig_once(ind_s))

    args_list=(indicator_df.loc[date] for date in dates) #fixme:
    ss=multi_process(ind2sig_once,args_list,15)


    ss=[s for s in ss if s is not None]
    if len(ss)>0:
        signal=pd.concat(ss,axis=1,sort=True).T
        return signal

def initialize():
    end='2018-08'
    fns=os.listdir(SINGLE_D_INDICATOR)
    names=[fn[:-4] for fn in fns]
    for name in names:
        signal_path=os.path.join(DIR_ROOT,'singleFactor','signal_pit',name+'.pkl')
        if os.path.exists(signal_path):
            pass
        else:
            indicator = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
            indicator=indicator[START:]
            signal=ind2sig_period(indicator,end=end)
            if signal is not None:
                signal.to_pickle(signal_path)
                print(name)

def update(date=None):
    fns = os.listdir(SINGLE_D_INDICATOR)
    names = [fn[:-4] for fn in fns][:2]  # fixme:
    for name in names:
        print(f'updating {name}')
        indicator = pd.read_pickle(
            os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
        signal_path = os.path.join(DIR_ROOT, 'singleFactor', 'signal_pit',
                                   name + '.pkl')
        signal_history=pd.read_pickle(signal_path)
        if date is None:
            for date,row in indicator[indicator.index>signal_history.index[-1]].iterrows():
                s=ind2sig_once(row)
                signal_history.loc[date]=s
        else:
            signal_history.loc[date]=ind2sig_once(indicator.loc[date])

        signal_history.to_pickle(signal_path)

if __name__ == '__main__':
    initialize()
    update()



#
# df=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\singleFactor\signal_pit\C__est_bookvalue_FT24M_to_close_chg_20.pkl')
#
# df.shape
# df=df.sort_index()
#

# if __name__ == '__main__':
#     convert_whole_history()


# df=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR,'T__idioVol_30.pkl'))
