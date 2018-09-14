# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-13  10:42
# NAME:FT_hp-test_consensus_time_sensitivity.py
from config import SINGLE_D_INDICATOR, DIR_TMP
import os
import pandas as pd
from singleFactor.combine_single_factors.combine_new import get_category, _grade

DIR_BACKTEST_SPANNING=os.path.join(DIR_TMP,'time_sensitivity')

fns=os.listdir(DIR_BACKTEST_SPANNING)

def get_hedged_ret(fn):
    s = pd.read_csv(
        os.path.join(DIR_BACKTEST_SPANNING, fn, 'hedged_returns.csv'),
        index_col=0, parse_dates=True, header=None).iloc[:, 0]
    s.name = fn
    return s

def get_sharpe(fn):
    return pd.read_csv(
            os.path.join(DIR_BACKTEST_SPANNING, fn, 'hedged_perf.csv'),
            index_col=0, parse_dates=True, header=None).iloc[:,0].loc['sharp_ratio']



def get_all_ret():
    df=pd.concat(multi_process(get_hedged_ret, fns, 20), axis=1)
    df.to_pickle(os.path.join(DIR_TMP,'df.pkl'))

def get_all_sharpe():
    fns = os.listdir(DIR_BACKTEST_SPANNING)
    sharpe=pd.Series(multi_process(get_sharpe,fns,n=20),index=fns)
    sharpe.to_pickle(os.path.join(DIR_TMP,'sharpe.pkl'))


def observe_top_performance():
    df=pd.read_pickle(os.path.join(DIR_TMP,'df.pkl'))
    sharpe=pd.read_pickle(os.path.join(DIR_TMP,'sharpe.pkl')).sort_values()
    targets=sharpe[sharpe>1.0].index.tolist()

    items=[]

    for col in targets:
        items.append((col,get_category(col)))

    for category in set([i[1] for i in items]):
        cols=[item[0] for item in items if category==item[1]][:5]
        print(category,len(cols))
        sub=df[cols]
        (1+sub).cumprod().plot().get_figure().savefig(os.path.join(DIR_TMP,f'{category}.png'))




import os
from backtest_zht.main_class import Backtest

from config import DIR_SIGNAL, DIR_BACKTEST_SPANNING
from tools import multi_process


SMOOTH_PERIODS=[0,10,20,30,40,50,60,70,80]

def derive_signal(raw_signal, smooth, sign):
    raw_signal= raw_signal * sign
    if smooth==0:
        return raw_signal
    else:
        return raw_signal.rolling(smooth, min_periods=int(smooth / 2)).mean()

def get_derive_signal(short_name,smooth,sign):
    raw_signal=pd.read_pickle(os.path.join(DIR_SIGNAL,short_name+'.pkl'))
    return derive_signal(raw_signal,smooth,sign)

def _bt_one_set(fn):
    raw_signal = pd.read_pickle(os.path.join(DIR_SIGNAL, fn))
    raw_signal=raw_signal.shift(10)#trick: lag 10 days
    for smooth in SMOOTH_PERIODS:
        for sign in [1, -1]:  # 1 denote positive, and -1 denotes negative
            name = '{}___smooth_{}___{}'.format(fn[:-4], smooth,
                                                {1: 'p', -1: 'n'}[sign])
            directory = os.path.join(DIR_TMP,'time_sensitivity',name)#fixme:
            signal=derive_signal(raw_signal,smooth,sign)
            Backtest(signal,name,directory)
            print(name)

def bt_all_spanning_signal():
    fns=os.listdir(DIR_SIGNAL)
    fns_c=[fn for fn in fns if fn.startswith('C_')]
    multi_process(_bt_one_set, fns_c, 20, multi_parameters=False) #fixme:
    # multiprocessing.Pool(20).map(_bt_one_set,fns)


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


def _read_ret(ln):
    _ret = pd.read_csv(
        os.path.join(DIR_BACKTEST_SPANNING, ln, 'hedged_returns.csv'),
        index_col=0, parse_dates=True, header=None)
    return (_ret, ln)

def get_strategy_ret():
    path=os.path.join(DIR_TMP,'d9316daf.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        long_names=os.listdir(DIR_BACKTEST_SPANNING)
        targets=[ln for ln in long_names if os.path.exists(os.path.join(DIR_BACKTEST_SPANNING,ln,'hedged_returns.csv'))]
        print(len(targets))
        ll=multi_process(_read_ret, targets)
        # ll=[_read_ret(tg) for tg in targets]
        ret=pd.concat([ele[0] for ele in ll],axis=1)
        ret.columns=[ele[1] for ele in ll]
        ret.to_pickle(path)
        return ret

def generate_signal_traverse():
    # grade=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\tmp\grade_strategy__M_500.pkl')
    # total number: 675
    ret=get_strategy_ret()
    short_window='200_iw3_cw3_5_criteria3'
    long_window='750_iw2_cw2_3_criteria3'

    effective_number=150
    signal_weight_mode=2

    window=750


    grade = grade_strategy(ret, freq='M', window=window)
    trd_dts = grade.index.get_level_values('trd_dt').unique().sort_values()[:-1]


    iw='iw3'
    cw='cw3'
    N=5
    criteria=3

    name=f'{window}_{iw}_{cw}_{N}_{criteria}'
    path=os.path.join(DIR_TMP,name)
    signal=get_signal(grade,criteria,N,trd_dts,iw,cw)
    signal.to_pickle(path)

#
# if __name__ == '__main__':
#     bt_all_spanning_signal()
