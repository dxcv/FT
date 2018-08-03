# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-03  19:11
# NAME:FT_hp-combine_new.py


import pandas as pd
import numpy as np


import os

from config import DIR_BACKTEST_SPANNING, DIR_TMP
from singleFactor.combine_single_factors.signal_spanning import \
    get_derive_signal
from tools import outlier, z_score, multi_task




'''
1. 变成每天调整
2. window,rolling expanding
3. 每类指标个数
4. 调整指标之间的加权方式以及大类之间的加权方式,signal,rank,equal,expotential decay
    IC,IC_IR....  https://zhuanlan.zhihu.com/p/31753606
5. 2009年开始回测
6. 选用不同的评分指标
7. effective number 由200变为100
8. cluster to get category rather than manually
9. PCA machine learning
10. 每天剔除失效的策略,加入新的策略，就如同选股股票调仓的方式，比如定义回撤超过20为失效的策略。
11. stop loss strategy
    Han, Y., Zhou, G., and Zhu, Y. (2016). Taming Momentum Crashes: A Simple Stop-Loss Strategy (Rochester, NY: Social Science Research Network).
12. constant volatility
13. ex post mean-variance efficient portfolios

'''



def get_category(ind_name):
    #TODO: cluster the factors,rather than classify them mannually

    if ind_name.startswith('V__'):
        return 'V'
    elif ind_name.startswith('T__turnover'):
        return 'turnover'
    elif ind_name.startswith('T__mom'):
        return 'mom'
    elif ind_name.startswith(('T__vol','T__beta','T__idio')):
        return 'vol'
    elif ind_name.startswith('Q_'):
        return 'Q'
    elif ind_name.startswith('G_'):
        return 'G'
    elif ind_name.startswith('C_'):
        return 'C'
    else:
        raise ValueError('No category for : {}'.format(ind_name))

def _read_ret(ln):
    try:
        _ret = pd.read_csv(
            os.path.join(DIR_BACKTEST_SPANNING, ln, 'hedged_returns.csv'),
            index_col=0, parse_dates=True, header=None)
        return (_ret,ln)
    except:
        pass


def get_strategy_ret():
    path=os.path.join(DIR_TMP,'ret.pkl')
    if os.path.exists(path):
        ret=pd.read_pickle(path)
    else:
        long_names=os.listdir(DIR_BACKTEST_SPANNING)
        ll=multi_task(_read_ret, long_names)
        ret=pd.concat([ele[0] for ele in ll],axis=1)
        ret.columns=[ele[1] for ele in ll]
        ret.to_pickle(os.path.join(DIR_TMP,'ret.pkl'))
    return ret

def _cum_ret(df):
    return ((1+df).cumprod()-1).iloc[-1,:]

def _return_down_ratio(x):
    def _for_series(s):
        value=(s+1).cumprod()
        ann_ret=value[-1]**(252/s.shape[0])-1
        max_drawdown=1-min(value/np.maximum.accumulate(value.fillna(-np.inf)))
        ret_down_ratio=ann_ret/max_drawdown
        return ret_down_ratio
    if isinstance(x,pd.Series):
        return _for_series(x)
    else:
        return x.apply(_for_series)

def _return_std_ratio(df):
    return df.mean()/df.std()

def _apply_rating_func(args):
    ret,window,day,rating_func=args
    sub = ret.loc[:day].iloc[-window:, :]
    # trick:only keep those indicators with enough history sample
    sub = sub.dropna(axis=1, thresh=int(sub.shape[0] * 0.95))
    ms=rating_func(sub)
    # ms = sub.apply(rating_func)
    return ms

def _grade(ret, rating_func,freq,window):
    days=ret.iloc[:,0].resample(freq).apply(lambda df:df.index[-1]).values#trick
    days=days[20:] # prune the header  #TODO:
    args_list=[(ret,window,d,rating_func) for d in days]

    mss=multi_task(_apply_rating_func, args_list)
    rating=pd.concat(mss,axis=1,keys=days)
    rating=rating.stack()
    rating.index.names=['long_name','trd_dt']
    return rating


rating_func_map={'criteria1':_cum_ret,'criteria2':_return_down_ratio,'criteria3':_return_std_ratio}


def grade_strategy(ret,freq,window):
    path=os.path.join(DIR_TMP,'grade_strategy__{}_{}.pkl'.format(freq,window))
    if os.path.exists(path):
        grade=pd.read_pickle(path)
    else:
        ratings=[]
        names=[]
        for name,func in rating_func_map.items():
            rt=_grade(ret,func,freq,window)
            ratings.append(rt)
            names.append(name)
        grade=pd.concat(ratings,axis=1,keys=names)
        grade['long_name']=grade.index.get_level_values('long_name')
        grade['short_name']=grade['long_name'].map(lambda x:x.split('___')[0])
        grade['smooth']=grade['long_name'].map(lambda x:int(x.split('___')[1].split('_')[1]))
        grade['sign']=grade['long_name'].map(lambda x:{'p':1,'n':-1}[x.split('___')[-1]])
        grade['category']=grade['long_name'].map(get_category)#TODO: cluster
        #TODO: log the shape of comb before and after .drop()
        grade=grade.dropna()
        grade.to_pickle(path)
    return grade

def standardize_signal(signal):
    '''
    Args:
        signal:DataFrame, panel

    Returns:DataFrame, panel,the shape may be different with the input dataframe

    '''
    stk=signal.stack()
    stk=stk.groupby('trd_dt').apply(outlier)
    stk=stk.groupby('trd_dt').apply(z_score)
    return stk.unstack()

def get_outer_frame(dflist):
    indsets=[set(df.index.tolist()) for df in dflist]
    colsets=[set(df.columns.tolist()) for df in dflist]
    indOuter=sorted(list(set.union(*indsets)))
    colOuter=sorted(list(set.union(*colsets)))
    return [df.reindex(index=indOuter,columns=colOuter) for df in dflist]

def get_month_data(grade,rb_dt,criteria,N):
    month_data= grade.loc[(slice(None), rb_dt), :]
    month_data = month_data.groupby(['trd_dt', 'short_name'],
                                  as_index=False, group_keys=False) \
        .apply(lambda df: df.loc[df[criteria].idxmax()])
    month_data=month_data[month_data[criteria] > 0] #only choose from
    month_data=month_data.groupby(['trd_dt', 'category'], as_index=False,
                                group_keys=False).apply(lambda df:df.nlargest(N,criteria))

    #indicator weight 1
    month_data['iw1']=1
    month_data['iw2']=month_data[criteria]
    month_data['iw3']=month_data.groupby('category')[criteria].rank()

    #category weight
    month_data['cw1']=1
    month_data['cw2']=month_data.groupby('category')[criteria].apply(lambda s:pd.Series([s.mean()] * len(s), index=s.index))
    s=month_data.groupby('category')[criteria].mean()
    r=month_data.groupby('category')[criteria].mean().rank()
    d=pd.Series(r.values,index=s.values).to_dict()
    month_data['cw3']=month_data['cw2'].replace(d)
    return month_data

def get_mixed_signal_of_next_month(month_data,rb_dt,next_rb_dt,iw,cw):
    month_data['weight']= month_data[iw] * month_data[cw]
    month_data['weight']/=month_data['weight'].sum()
    frags=[]
    ws=[]
    for _,row in month_data.iterrows():#TODO: use multiProcessing
        short_name=row.loc['short_name']
        smooth=row.loc['smooth']
        sign=row.loc['sign']
        frag = get_derive_signal(short_name, smooth, sign)[rb_dt:next_rb_dt][1:]
        if frag.shape[0]>0:
            frags.append(frag)
            ws.append(row.loc['weight'])

    signals=[standardize_signal(fr) for fr in frags]
    signals=get_outer_frame(signals)
    signals = [s * w for s, w in zip(signals, ws)]
    mixed = pd.DataFrame(np.nanmean([s.values for s in signals], axis=0),
                         index=signals[0].index, columns=signals[0].columns)
    return mixed

def _mix_one_slice(args):
    grade,criteria,trd_dts,i,N=args
    rb_dt = trd_dts[i]  # rebalance date
    print(rb_dt)
    next_rb_dt = trd_dts[i + 1]  # next rebalance date
    month_data = get_month_data(grade, rb_dt, criteria, N)
    iw = 'iw2'
    cw = 'cw3'
    mixed_signal = get_mixed_signal_of_next_month(month_data, rb_dt, next_rb_dt,
                                                  iw, cw)
    return mixed_signal
    # mixed_signal_frags.append(mixed_signal)


def get_signal(grade,criteria,N,trd_dts):
    args_list=[(grade,criteria,trd_dts,i,N) for i in range(len(trd_dts)-1)]
    mixed_signal_frags=multi_task(_mix_one_slice,args_list,20)
    signal=pd.concat(mixed_signal_frags)
    return signal

def generate_signal():
    # grade=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\tmp\grade_strategy__M_500.pkl')
    ret=get_strategy_ret()
    grade=grade_strategy(ret,freq='M',window=500)#fixme:
    criteria = 'criteria1'
    N = 5
    trd_dts= grade.index.get_level_values('trd_dt').unique().sort_values()[:-1]
    signal=get_signal(grade,criteria,N,trd_dts)
    signal.to_pickle(os.path.join(DIR_TMP,'signal.pkl'))


def debug():
    ret=get_strategy_ret()
    grade=grade_strategy(ret,freq='M',window=500)
    criteria = 'criteria1'
    N = 5
    trd_dts = grade.index.get_level_values('trd_dt').unique().sort_values()[:-1]
    args_list=[(grade,criteria,trd_dts,i,N) for i in range(len(trd_dts)-1)]
    frags=[]
    for args in args_list:
        print(args[3])
        frag=_mix_one_slice(args)
        frags.append(frag)


if __name__ == '__main__':
    generate_signal()
    # debug()
