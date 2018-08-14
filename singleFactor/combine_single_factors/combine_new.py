# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-03  19:11
# NAME:FT_hp-combine_new.py


import pandas as pd
import numpy as np


import os

from backtest_zht.main_class import Backtest, DEFAULT_CONFIG
from config import DIR_BACKTEST_SPANNING, DIR_TMP, DIR_MIXED_SIGNAL_BACKTEST, \
    DIR_MIXED_SIGNAL
from singleFactor.combine_single_factors.signal_spanning import \
    get_derive_signal
from singleFactor.combine_single_factors.summary import summarize
from tools import outlier, z_score, multi_task




'''
1. 变成每天调整
2. window,rolling expanding
3. 每类指标个数
4. 调整指标之间的加权方式以及大类之间的加权方式,signal,rank,equal,expotential decay
    IC,IC_IR....  https://zhuanlan.zhihu.com/p/31753606
    quantile as weight
    use regression to combine different indicators (just like Fama macbeth regression)
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
14. 之所以在2017年之后表现不好，是因为我们选的因子都是从研报里来的，这些因子都已经被证明
    在研报发布以前表现良好，在2017年和2018年，实际这些因子相当于是做样本外测试，因为
    在2017年之后可能，由于有些机构开始大规模的使用这些因子，使得他们失效了。如何实证这些因子
    确实已经被使用到了市场？

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

    # mss=multi_task(_apply_rating_func, args_list) #fixme:
    mss=[]
    for args in args_list:
        s=_apply_rating_func(args)
        mss.append(s)

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
    #select the best smoothing period and direction
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
    #TODO: this function is really slow
    month_data['weight']= month_data[iw] * month_data[cw]
    month_data['weight']/=month_data['weight'].sum()
    frags=[]
    ws=[]
    for _,row in month_data.iterrows():#TODO: use multiProcessing
        short_name=row.loc['short_name']
        smooth=row.loc['smooth']
        sign=row.loc['sign']
        '''
        .copy() to save momery, if .copy() is not called, the whole DataFrame will
        be stored in RAM, but we just need to use part of them (.loc[rb_dt:next_rb_dt][1:]).
        If .copy() is used, only those indexed part will be stored in RAM.
        '''
        frag = get_derive_signal(short_name, smooth, sign).loc[rb_dt:next_rb_dt][1:].copy()
        if frag.shape[0]>0:
            frags.append(frag)
            ws.append(row.loc['weight'])
    signals=[standardize_signal(fr) for fr in frags]#TODO: too slow
    signals=get_outer_frame(signals)
    signals = [s * w for s, w in zip(signals, ws)]
    mixed = pd.DataFrame(np.nanmean([s.values for s in signals], axis=0),
                         index=signals[0].index, columns=signals[0].columns)
    return mixed

def _mix_one_slice(args):
    grade,criteria,trd_dts,i,N,iw,cw=args
    rb_dt = trd_dts[i]  # rebalance date
    print(rb_dt)
    next_rb_dt = trd_dts[i + 1]  # next rebalance date
    month_data = get_month_data(grade, rb_dt, criteria, N)
    mixed_signal = get_mixed_signal_of_next_month(month_data, rb_dt, next_rb_dt,
                                                  iw, cw)
    return mixed_signal
    # mixed_signal_frags.append(mixed_signal)

def gen_args(grade,criteria,N,trd_dts,iw,cw):#trick: use this way to save memoery,
    for i in range(len(trd_dts)-1):#fixme:
    # for i in range(len(trd_dts)-5,len(trd_dts)-4):
        yield (grade,criteria,trd_dts,i,N,iw,cw)

def get_signal_debug(grade,criteria,N,trd_dts,iw,cw):
    args_generator = gen_args(grade, criteria, N, trd_dts, iw, cw)
    signal=pd.concat([_mix_one_slice(args) for args in args_generator])
    return signal


def get_signal(grade,criteria,N,trd_dts,iw,cw):
    # args_list=[(grade,criteria,trd_dts,i,N,iw,cw) for i in range(len(trd_dts)-1)]
    args_generator=gen_args(grade,criteria,N,trd_dts,iw,cw)
    mixed_signal_frags=multi_task(_mix_one_slice,args_generator,30)
    signal=pd.concat(mixed_signal_frags)
    return signal

def generate_signal():
    # grade=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\tmp\grade_strategy__M_500.pkl')
    # total number: 675
    ret=get_strategy_ret()
    for window in [1000,500,200,50]:
        grade=grade_strategy(ret,freq='M',window=window)#fixme:
        trd_dts = grade.index.get_level_values('trd_dt').unique().sort_values()[:-1]
        for iw in ['iw1','iw2','iw3']:
            for cw in ['cw1','cw2','cw3']:
                for N in [1, 3, 5, 10, 100]:
                    for criteria in ['criteria1','criteria2','criteria3']:
                        name='{}_{}_{}_{}_{}.pkl'.format(window,iw,cw,N,criteria)
                        path=os.path.join(DIR_MIXED_SIGNAL,name)
                        if not os.path.exists(path):
                            signal=get_signal(grade,criteria,N,trd_dts,iw,cw)
                            signal.to_pickle(path)
                        print(name)

def debug_generate_signal():
    ret=get_strategy_ret()
    for window in [200,50]:
        grade=grade_strategy(ret,freq='M',window=window)#fixme:
        trd_dts = grade.index.get_level_values('trd_dt').unique().sort_values()[:-1]
        for iw in ['iw1','iw2','iw3']:
            for cw in ['cw1','cw2','cw3']:
                for N in [1, 3, 5, 10, 100]:
                    for criteria in ['criteria1','criteria2','criteria3']:
                        name='{}_{}_{}_{}_{}.pkl'.format(window,iw,cw,N,criteria)
                        path=os.path.join(DIR_MIXED_SIGNAL,name)
                        if not os.path.exists(path):
                            signal=get_signal(grade,criteria,N,trd_dts,iw,cw)
                            signal.to_pickle(path)
                        print(name)

def debug():
    ret=get_strategy_ret()
    freq='M'
    window=200
    ratings = []
    names = []
    for name, func in rating_func_map.items():
        days = ret.iloc[:, 0].resample(freq).apply(
            lambda df: df.index[-1]).values  # trick
        days = days[20:]  # prune the header  #TODO:
        args_list = [(ret, window, d, func) for d in days]
        mss=[]
        for args in args_list:
            print(args[1],args[2],name)
            m=_apply_rating_func(args)
            mss.append(m)
        rating = pd.concat(mss, axis=1, keys=days)
        rating = rating.stack()
        rating.index.names = ['long_name', 'trd_dt']
        ratings.append(rating)
        names.append(name)
    grade = pd.concat(ratings, axis=1, keys=names)
    grade['long_name'] = grade.index.get_level_values('long_name')
    grade['short_name'] = grade['long_name'].map(lambda x: x.split('___')[0])
    grade['smooth'] = grade['long_name'].map(
        lambda x: int(x.split('___')[1].split('_')[1]))
    grade['sign'] = grade['long_name'].map(
        lambda x: {'p': 1, 'n': -1}[x.split('___')[-1]])
    grade['category'] = grade['long_name'].map(get_category)  # TODO: cluster
    # TODO: log the shape of comb before and after .drop()
    grade = grade.dropna()

    # criteria = 'criteria1'
    # N = 1
    # iw='iw1'
    # cw='cw1'
    # trd_dts = grade.index.get_level_values('trd_dt').unique().sort_values()[:-1]
    # args_list=[(grade,criteria,trd_dts,i,N,iw,cw) for i in range(len(trd_dts)-1)]
    # frags=[]
    # for args in args_list:
    #     print(args[3])
    #     frag=_mix_one_slice(args)
    #     frags.append(frag)


def _task_bt(fn):
    signal = pd.read_pickle(os.path.join(DIR_MIXED_SIGNAL, fn))
    for effective_number in [100, 150, 200, 300]:
        for signal_weight_mode in [1, 2, 3]:
            name = '{}_{}_{}'.format(fn[:-4], effective_number,signal_weight_mode)
            directory = os.path.join(DIR_MIXED_SIGNAL_BACKTEST, name)

            cfg = DEFAULT_CONFIG
            cfg['effective_number'] = effective_number
            cfg['signal_to_weight_mode'] = signal_weight_mode
            Backtest(signal, name=name, directory=directory, start='2009',
                     config=cfg)

def debug1():
    fn='1000_iw2_cw1_100_criteria1.pkl'
    _task_bt(fn)



def backtest_mixed_signal():
    fns=os.listdir(DIR_MIXED_SIGNAL)
    multi_task(_task_bt,fns,30)




# if __name__ == '__main__':
#
#     debug1()



if __name__ == '__main__':
    # generate_signal()
    debug_generate_signal()
    # backtest_mixed_signal()
    # summarize()
