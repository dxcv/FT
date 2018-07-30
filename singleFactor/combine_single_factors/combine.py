# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-29  09:27
# NAME:FT_hp-combine.py

import multiprocessing
from contextlib import closing

from backtest_zht.main import run_backtest
from backtest_zht.main_class import Backtest
from config import DIR_TMP, DIR_BACKTEST_SPANNING
import os
import pandas as pd
import numpy as np

from singleFactor.combine_single_factors.signal_spanning import \
    get_derive_signal
from tools import outlier, z_score

'''
1. 指标的切换频率由每年调整变为每月调整
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


def _multi_task(func,args_list,n=30):
    pool=multiprocessing.Pool(n)
    results=pool.map(func,args_list)
    pool.close()#trick: close the processing every time the pool has finished its task, and pool.close() must be called before pool.join()
    pool.join()
    #refer to https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    return results

def get_strategy_ret():
    long_names=os.listdir(DIR_BACKTEST_SPANNING)
    ll=_multi_task(_read_ret,long_names)
    ret=pd.concat([ele[0] for ele in ll],axis=1)
    ret.columns=[ele[1] for ele in ll]
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

    mss=_multi_task(_apply_rating_func,args_list)
    # with closing(multiprocessing.Pool(30)) as p:
    #     mss=p.map(_apply_rating_func,args_list)

    # pool=multiprocessing.Pool(30)
    # mss=pool.map(_apply_rating_func,args_list)
    # pool.close()
    # pool.join()

    # mss=multiprocessing.Pool(30).map(_apply_rating_func, args_list)
    rating=pd.concat(mss,axis=1,keys=days)
    rating=rating.stack()
    rating.index.names=['long_name','trd_dt']
    return rating

def grade_strategy(ret,freq,window):
    ratings=[]
    for rating_func in [_cum_ret,_return_down_ratio,_return_std_ratio]:
        rt=_grade(ret, rating_func,freq,window)
        ratings.append(rt)
    comb=pd.concat(ratings,axis=1,keys=['cumprod_ret','return_down_ratio','return_std_ratio'])

    # r1=_grade(ret,_cum_ret,freq,window)
    # r2=_grade(ret,_return_down_ratio,freq,window)
    # r3=_grade(ret,_return_std_ratio,freq,window)
    # comb=pd.concat([r1,r2,r3],axis=1,keys=['cumprod_ret','return_down_ratio','return_std_ratio'])


    comb['long_name']=comb.index.get_level_values('long_name')
    comb['short_name']=comb['long_name'].map(lambda x:x.split('___')[0])
    comb['smooth']=comb['long_name'].map(lambda x:int(x.split('___')[1].split('_')[1]))
    comb['sign']=comb['long_name'].map(lambda x:{'p':1,'n':-1}[x.split('___')[-1]])
    comb['category']=comb['long_name'].map(get_category)#TODO: cluster
    #TODO: log the shape of comb before and after .drop()
    comb=comb.dropna()
    return comb

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

def get_mixed_signals(signals,weight=None):
    signals=[standardize_signal(signal) for signal in signals]
    signals = get_outer_frame(signals)
    if weight is None:
        mixed = pd.DataFrame(np.nanmean([s.values for s in signals], axis=0),
                             index=signals[0].index, columns=signals[0].columns)
    else:
        raise NotImplementedError
    return mixed

def select_strategies(comb, criteria, num_per_category):
    #select the best smooth window
    comb=comb.groupby(['trd_dt', 'short_name'],
                      as_index=False, group_keys=False)\
        .apply(lambda df:df.loc[df[criteria].idxmax()])

    comb=comb.groupby(['trd_dt', 'category'],
                      as_index=False, group_keys=False).apply(
        lambda df:df.nlargest(num_per_category, criteria))
    comb.drop('short_name', axis=1, inplace=True)
    table=comb.reset_index()

    # table.to_pickle(os.path.join(DIR_TMP,'table.pkl'))
    return table

def _one_slice(args):
    table,trd_dts,i=args
    rb_dt = trd_dts[i]  # rebalance date
    next_rb_dt = trd_dts[i + 1]  # next rebalance date
    sub = table[table['trd_dt'] == rb_dt]
    y_signal_l = []
    for c in sub['category'].unique():
        ss = sub[sub['category'] == c]
        c_signal_l = []
        for _, row in ss.iterrows():
            short_name = row.loc['short_name']
            smooth = row.loc['smooth']
            sign = row.loc['sign']
            # trick:do not include the trd_dt
            c_signal_l.append(get_derive_signal(short_name, smooth, sign)[
                              rb_dt:next_rb_dt][1:])
        c_signal = get_mixed_signals(c_signal_l)
        y_signal_l.append(c_signal)
    y_signal = get_mixed_signals(y_signal_l)
    print(i,rb_dt)
    return y_signal

def generate_signal(table):
    trd_dts=table['trd_dt'].unique()
    args_list=[(table,trd_dts,i) for i in range(len(trd_dts[:-1]))]
    signal_monthly=_multi_task(_one_slice,args_list)
    # signal_monthly=multiprocessing.Pool(30).map(_one_slice,args_list)
    comb_signal = pd.concat(signal_monthly)
    return comb_signal

def gen_config(window=500,num_per_category=5,criteria='cumprod_ret',effective_number=200):
    config = {
        'freq': 'M',
        'window': 500,
        'num_per_category': 5,
        'criteria': 'cumprod_ret',

        'effective_number': 100,
        'target_number': 100,
        'signal_to_weight_mode': 3,  # 等权重
        # 'decay_num': 1,　＃TODO：　used to smooth the signal
        # 'delay_num': 1,
        'hedged_period': 60,
        # trick: 股指的rebalance时间窗口，也可以考虑使用风险敞口大小来作为relance与否的依据
        'buy_commission': 2e-3,
        'sell_commission': 2e-3,
        'tax_ratio': 0.001,  # 印花税
        'capital': 10000000,  # 虚拟资本,没考虑股指期货所需要的资金
    }

    config['window']=window
    config['num_per_category']=num_per_category
    config['criteria']=criteria
    config['effective_number']=effective_number
    return config

def gen_comb_signal():
    configs=[]
    for window in [50,100,300,500,1000]:
        for num_per_category in [1,3,5,10]:
            for criteria in ['cumprod_ret','return_down_ratio','return_std_ratio']:
                for effective_number in [100,150,200,300]:
                    config=gen_config(window,num_per_category,criteria,effective_number)
                    configs.append(config)


    for config in configs:
        name = '{}_{}_{}_{}'.format(
                                    config['window'],
                                    config['num_per_category'],
                                    config['criteria'],
                                    config['effective_number'])
        print(name,'starting')
        ret = get_strategy_ret()
        comb = grade_strategy(ret, config['freq'], config['window'])

        # comb=pd.read_pickle(os.path.join(DIR_TMP,'comb.pkl'))

        table = select_strategies(comb, config['criteria'],
                                  config['num_per_category'])
        comb_signal = generate_signal(table)

        # comb_signal=pd.read_pickle(os.path.join(DIR_TMP,'signal_{}.pkl'.format(name)))

        comb.to_pickle(os.path.join(DIR_TMP,'comb_{}.pkl'.format(name)))
        comb_signal.to_pickle(os.path.join(DIR_TMP,'signal_{}.pkl'.format(name)))


def debug(config):
    name = '{}_{}_{}_{}'.format(
                                config['window'],
                                config['num_per_category'],
                                config['criteria'],
                                config['effective_number'])

    ret = get_strategy_ret()
    comb = grade_strategy(ret, config['freq'], config['window'])

    # comb=pd.read_pickle(os.path.join(DIR_TMP,'comb.pkl'))

    table = select_strategies(comb, config['criteria'],
                              config['num_per_category'])
    comb_signal = generate_signal(table)

    # comb_signal=pd.read_pickle(os.path.join(DIR_TMP,'signal_{}.pkl'.format(name)))

    comb.to_pickle(os.path.join(DIR_TMP,'comb_{}.pkl'.format(name)))
    comb_signal.to_pickle(os.path.join(DIR_TMP,'signal_{}.pkl'.format(name)))



    # Backtest(comb_signal, name, directory=os.path.join(DIR_TMP, name),
    #          start='2009', config=config)


def run():
    config1=gen_config()
    config2=gen_config(window=300)
    config3=gen_config(num_per_category=2)
    config4=gen_config(criteria='return_down_ratio') #'cumprod_ret','return_down_ratio','return_std_ratio'
    config5=gen_config(effective_number=300)


    for config in [config1,config2,config3,config4,config5]:
        print('start',config)
        debug(config)
        print('end:',config)


def debug1():
    config4=gen_config(criteria='return_down_ratio') #'cumprod_ret','return_down_ratio','return_std_ratio'
    debug(config4)

def _task(args):
    signal,name,directory,config=args
    Backtest(signal,name=name,directory=directory,start='2009',config=config)


def debug2():
    fns=[fn for fn in os.listdir(os.path.join(DIR_TMP)) if fn.startswith('signal_') and fn.endswith('.pkl')]

    args_list=[]
    for fn in fns:
        print(fn)
        signal = pd.read_pickle(os.path.join(DIR_TMP,fn))
        window,num_per_category,criteria,effective_number=int(fn.split('_')[1]),int(fn.split('_')[2]),'_'.join(fn.split('_')[3:-1]),int(fn.split('_')[-1][:-4])
        cfg=gen_config(window,num_per_category,criteria,effective_number)
        name=fn[:-4]
        args_list.append((signal,name,os.path.join(DIR_TMP,name),cfg))

    # _multi_task(_task,args_list,n=20)


        Backtest(signal,name=name,directory=os.path.join(DIR_TMP,name),start='2009',config=cfg)




if __name__ == '__main__':
    debug2()



# if __name__ == '__main__':
#     debug1()

# if __name__ == '__main__':
#     gen_comb_signal()


