# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  10:57
# NAME:FT-utils.py
import multiprocessing
import time
from itertools import chain, islice
from multiprocessing.pool import ThreadPool

import pandas as pd
from multiprocessing import Process
from pandas.tseries.offsets import MonthEnd
import numpy as np

from scipy import stats
import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
import statsmodels.api as sm

def monitor(func):
    def wrapper(*args,**kwargs):
        print('{}   starting -> {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),
                                           func.__name__))
        return func(*args,**kwargs)
    return wrapper

def get_inter_index(s1, s2):
    interInd=s1.index.intersection(s2.index)
    s1=s1.reindex(interInd)
    s2=s2.reindex(interInd)
    return s1, s2

def _convert_freq(x,freq,thresh):
    x=x.groupby(pd.Grouper(freq=freq,level='trd_dt')).last()
    # TODO: ffill whould only be used on indicators from financial report.
    # TODO: pay attention to cash_div and counts in
    x=x.ffill(limit=thresh)
    return x

def convert_freq(x, freq='M', thresh=12):
    newdf=x.groupby('stkcd').apply(_convert_freq, freq, thresh)
    newdf=newdf.swaplevel().sort_index()
    return newdf

def handle_duplicates(df):
    return df[~df.index.duplicated(keep='first')]

def number2dateStr(x):
    if x:
        if isinstance(x,(int,float)):
            x=str(x)

        if '.' in x:
            x=x.split('.')[0]
        return x

def daily2monthly(df):
    '''

    Args:
        df:DataFrame,contains column ['stkcd','trd_dt']

    Returns:DataFrame, only add a new column named 'month_end' to the input df

    '''
    df=df.sort_values(['stkcd','trd_dt'])
    monthly=df[(df['stkcd']==df['stkcd'].shift(-1)) &
                     (df['trd_dt'].dt.month!=df['trd_dt'].shift(-1).dt.month)]
    monthly=monthly.dropna(how='all')
    monthly['month_end']=monthly['trd_dt']+MonthEnd(0)
    return monthly

def filter_st_and_young(df,fdmt_m):
    data=pd.concat([fdmt_m,df],axis=1).reindex(fdmt_m.index)
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    return data

def outlier(s, k=4.5):
    '''
    Parameters
    ==========
    s:Series
        原始因子值
    k = 3 * (1 / stats.norm.isf(0.75))
    '''
    med = np.median(s)
    '''
    trick: NaN should be removed before apply this function,
    to skip this warning, we can stack the panel dataframe and 
    then use groupby().apply(), since stacking procedure will 
    dropna by default.
    '''
    mad = np.median(np.abs(s - med))
    uplimit = med + k * mad
    lwlimit = med - k * mad
    y = np.where(s >= uplimit, uplimit, np.where(s <= lwlimit, lwlimit, s))
    # return pd.DataFrame(y, index=s.index)
    return pd.Series(y, index=s.index)

def z_score(x):
    return (x - np.mean(x)) / np.std(x)

def neutralize(df, col, industry, cap='ln_cap'):
    '''
    Parameters
    ===========
    df:
        包含标准化后的因子值的DataFrame
    industry: list of industry columns
        排除第一行业代码后的m-1个行业代码

    Returns
    =======
    res:
        标准化因子对行业哑变量矩阵和对数市值回归后的残差
    '''
    a = np.array(df.loc[:, industry + [cap]])
    A = np.hstack([a, np.ones([len(a), 1])])
    y = df.loc[:, col]
    beta = np.linalg.lstsq(A, y,rcond=-1)[0] #fixme: rcond=None?
    res = y - np.dot(A, beta)
    return res


def convert_indicator_to_signal(df, name):
    '''

    Args:
        df: DataFrame with multiIndex
        name: col to apply

    Returns:

    '''
    df[name]=df[name].groupby('month_end').apply(outlier)
    df[name]=df[name].groupby('month_end').apply(z_score)
    return df


def clean(df, col,by='month_end'):
    '''
    filter out abnormal value
    s-score
    industry-neutralized
    log_cap neutralized

    Parameters
    ==========
    df: DataFrame
        含有因子原始值、市值、行业代码
    col:
        因子名称
    '''

    # Review: 风格中性：对市值对数和市场做回归后取残差
    #TODO： 市值中性化方式有待优化，可以使用SMB代替ln_cap
    df[col + '_out']=df.groupby(by)[col].apply(outlier) #trick: dropna before applying function outlier
    df[col + '_zsc']=df.groupby(by)[col + '_out'].apply(z_score)
    df['wind_2'] = df['wind_indcd'].apply(str).str.slice(0, 6) # wind 2 级行业代码
    df = df.join(pd.get_dummies(df['wind_2'], drop_first=True))
    df['ln_cap'] = np.log(df['cap'])
    industry = list(np.sort(df['wind_2'].unique()))[1:]
    df[col + '_neu'] = df.groupby(by, group_keys=False).apply(neutralize, col + '_zsc', industry)

    del df[col]
    del df[col + '_out']
    del df[col + '_zsc']
    df=df.rename(columns={col + '_neu':col})
    return df


def myroll(df, d):
    '''
    refer to
        https://stackoverflow.com/questions/39501277/efficient-python-pandas-stock-beta-calculation-on-many-dataframes
    '''

    # stack df.values d-times shifted once at each stack
    roll_array = np.dstack([df.values[i:i + d, :] for i in range(len(df.index) - d + 1)]).T
    # roll_array is now a 3-D array and can be read into
    # a pandas panel object
    panel = pd.Panel(roll_array,
                     items=df.index[d - 1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(d), name='roll'))
    # convert to dataframe and pivot + groupby
    # is now ready for any action normally performed
    # on a groupby object
    #trick: filter_obsservations=False
    return panel.to_frame(filter_observations=False).unstack().T.groupby(level=0)

def chunks(iterable, size=500):
    '''chunks an iterable into n smaller generator with the given maximium size for each generator,
    The function will keep the initial order.

    refer to https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it
    '''
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def _multi_p(func,args_iter,n,multi_parameters):
    '''
        make sure that all the data needed by "func" should be sent by parameters, try to avoid calling
        the data outside the "func" since sometimes the processes may be frozen without raising any
        error.

        Args:
            func:
            args_iter:
            n:
            multi_parameters:

        Returns:

        '''
    pool = multiprocessing.Pool(n)
    if multi_parameters:
        results = pool.starmap(func, args_iter)
    else:
        results = pool.map(func, args_iter)
    pool.close()  # trick: close the processing every time the pool has finished its task, and pool.close() must be called before pool.join()
    pool.join()
    # refer to https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    return results

def multi_process(func, args_iter, n=15, multi_parameters=False, size_in_each_group=None):
    '''

    Args:
        func:
        args_iter:
        n:
        multi_parameters:
        size_in_each_group:None or int, the times of the task being called in each _multi_p function,
        by default it is None. This parameters can be set in case "_multi_p" will consume more and
        more memory when the 'task' is called over and over again. With 'size_in_each_group" set
        '_multi_p' will be ended and start again each time it has finished the given number of task to save momery.


    Returns:

    '''
    if size_in_each_group is None:
        return _multi_p(func, args_iter, n, multi_parameters)
    else:
        rs=[]
        for sub_args_iter in chunks(args_iter, size_in_each_group):
            rs+=_multi_p(func,sub_args_iter,n,multi_parameters)
        return rs

def multi_process_old(func, args_iter, n=20):
    pool=multiprocessing.Pool(n)
    results=pool.map(func, args_iter)
    pool.close()#trick: close the processing every time the pool has finished its task, and pool.close() must be called before pool.join()
    pool.join()
    #refer to https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    return results

def run_in_parallel(funcs):
    '''
    run multiple functions in parallel
    Args:
        funcs: a list of function

    Returns:

    '''
    proc=[]
    for f in funcs:
        p=Process(target=f)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

def multi_thread(func,args_iter,n=50,multi_parameters=False):
    if multi_parameters:
        results=ThreadPool(n).starmap(func,args_iter)
    else:
        results=ThreadPool(n).map(func,args_iter)
    return results

def mytiming(func):
    def wrapper(*args,**kwargs):
        t1 = time.time()
        result=func(*args,**kwargs)
        t2=time.time()
        print(f'{func.__name__}----{t2-t1} seconds')
        return result
    return wrapper

#--------------------------------------------------------------------------
def premium_ic(x, y, f, retn, t, s):
    '''
        Parameters
    ==========
    x:
        包含因子值、收益率的DataFrame
    y: list
        period
    s: int
        时期间隔

    Returns
    =======
    retn_(t + s) = a + b * f_t # 用下一期收益率都当期因子进行OLS回归
    '''
    bp_t0 = x[f][y[t]]
    rt_tn = x[retn][y[t + s]]
    bp_rt = pd.concat([bp_t0, rt_tn], axis=1).dropna()
    sl = stats.linregress(bp_rt[f], bp_rt[retn])
    beta = sl.slope
    tvalue = sl.slope / sl.stderr  # trick: tvalue = mu / stderr, stderr = std / sqrt(n-1)
    ic_value = stats.spearmanr(bp_rt)[0]
    btic = pd.DataFrame({'beta': [beta], 'tvalue': [tvalue], 'ic': [ic_value]},
                        columns=['beta', 'tvalue', 'ic'])
    return btic


def premium2_ic(x, y, f, retn, t, s, w='cap'):  # numpy.linalg求解WLS
    '''
    Returns
    =======
    retn = a + b * f # 用下一期收益率都当期因子进行以为权重的WLS回归
    相比于OLS回归并没有减小异方差
    '''
    bp_t0 = x[f][y[t]]
    rt_tn = x[retn][y[t + s]]
    size = x[w][y[t]]
    bp_rt = pd.concat([bp_t0, rt_tn, size], axis=1).dropna()
    a = np.array(bp_rt[f])
    A = np.vstack([a, np.ones(len(a))]).T
    B = np.array(bp_rt[retn])
    w_inv = np.diag(np.sqrt(bp_rt[w]))  # 市值平方根作为权重
    AT_w_inv = np.dot(A.T, w_inv)
    beta = np.linalg.inv(np.dot(AT_w_inv, A)).dot(AT_w_inv).dot(B)
    sse = np.sum((B - A.dot(beta)) ** 2)
    tvalue = beta[0] * sqrt(len(a)) * np.std(a) / sqrt(sse / (len(a) - 2))
    ic_value = stats.spearmanr(bp_rt.loc[:, [f, retn]])[0]
    btic = pd.DataFrame(
        {'beta': [beta[0]], 'tvalue': [tvalue], 'ic': [ic_value]},
        columns=['beta', 'tvalue', 'ic'])
    return btic


def premium3_ic(x, y, f, retn, t, s, w='cap'):  # sm.WLS求解
    '''
    Returns
    =======
    retn = a + b * f # 用下一期收益率都当期因子进行以为权重的WLS回归
    相比于OLS回归并没有减小异方差
    '''
    bp_t0 = x[f][y[t]]
    rt_tn = x[retn][y[t + s]]
    size = x[w][y[t]]
    bp_rt = pd.concat([bp_t0, rt_tn, size], axis=1).dropna()
    X = sm.add_constant(bp_rt[f])
    Y = bp_rt[retn]
    wls = sm.WLS(Y, X, weights=bp_rt[w]).fit()
    beta = wls.params[0]
    tvalue = wls.tvalues[0]
    ic_value = stats.spearmanr(bp_rt.loc[:, [f, retn]])[0]
    btic = pd.DataFrame({'beta': [beta], 'tvalue': [tvalue], 'ic': [ic_value]},
                        columns=['beta', 'tvalue', 'ic'])
    return btic


def risk_premium(x, y, f, retn, s=1, w=None):
    '''
    Parameters
    ==========
    x:
        包含因子值、收益率的DataFrame
    y: list
        period
    s: int
        时期间隔

    Returns
    =======
    return premium_result, btic, ic # 返回一个元组
    premium_reuslt:
        风险溢价、T值、IC值的统计数据
    btic:
        风险溢价和T值序列
    ic:
        IC值序列
    '''

    if w == None:
        btic = pd.concat({y[t]: premium_ic(x, y, f, retn, t, s) \
                          for t in range(len(y) - s)}, axis=0)
    else:
        btic = pd.concat({y[t]: premium2_ic(x, y, f, retn, t, s, w='cap') \
                          for t in range(len(y) - s)}, axis=0)
    btic.index = y[:len(y) - s]
    premium_result = {'Factor return mean ': btic.beta.mean(),
                      'Factor return std': btic.beta.std(),
                      'P(t > 0)': len(btic[btic.tvalue > 0]) / len(btic),
                      'P(|t| > 2)': len(btic[abs(btic.tvalue) > 2]) / len(btic),
                      '|t| mean': abs(btic.tvalue).mean(),
                      'IC mean': btic.ic.mean(),
                      'IC std': btic.ic.std(),
                      'P(IC > 0)': len(btic[btic.ic > 0]) / len(btic.ic),
                      'P(IC > 0.02)': len(btic[btic.ic > 0.02]) / len(btic.ic),
                      'IR of IC': btic.ic.mean() / btic.ic.std()}
    premium_result = pd.DataFrame(
        premium_result, \
        columns=['Factor return mean ', 'Factor return std', 'P(t > 0)',
                 'P(|t| > 2)', '|t| mean', 'IC mean', 'IC std', 'P(IC > 0)',
                 'P(IC > 0.02)', 'IR of IC'],
        index=[f]).T
    return premium_result, btic


def heter(x, y, t, f, retn, w='cap', s=1):
    '''
    Returns
    =======
    retn = a + b * f # 用下一期收益率都当期因子进行回归
    检验OLS下的残差是否存在异方差
    返回散点图
    '''
    bp_t0 = x[f][y[t]]
    rt_tn = x[retn][y[t + s]]
    size = x[w][y[t]]
    out = size.mean() + 2 * size.std()
    size[size > out] = out
    bp_rt = pd.concat([bp_t0, rt_tn, size], axis=1).dropna()
    sl = stats.linregress(bp_rt[f], bp_rt[retn])
    beta = sl.slope
    bp_rt['error'] = bp_rt[retn] - sl.intercept - beta * bp_rt[f]

    #    plt.figure(figsize=(5, 5))
    #    plt.scatter(bp_rt[f] ** 2, bp_rt['error'])
    #    plt.xlabel(f)
    #    plt.ylabel('error')
    #    plt.title('Residual of ' + y[t])

    plt.figure(figsize=(5, 5))
    out = bp_rt[w].mean() + 2 * bp_rt[w].std()
    bp_rt[w][bp_rt[w] > out] = out
    plt.scatter(np.sqrt(bp_rt[w]), bp_rt['error'])
    plt.xlabel('sqrt(' + w + ')')
    plt.ylabel('error')
    plt.title('Residual of ' + y[t])


def heter2(x, y, t, f, retn, w, s=1):
    '''
    Returns
    =======
    retn = a + b * f # 用下一期收益率都当期因子进行回归
    检验WLS下的残差是否存在异方差
    返回散点图
    '''
    bp_t0 = x[f][y[t]]
    rt_tn = x[retn][y[t + s]]
    size = x[w][y[t]]
    bp_rt = pd.concat([bp_t0, rt_tn, size], axis=1).dropna()
    a = np.array(bp_rt[f])
    n = len(a)
    A = np.vstack([a, np.ones(n)]).T
    B = np.array(bp_rt[retn])
    w_inv = np.diag(np.sqrt(bp_rt[w]))
    AT_w_inv = np.dot(A.T, w_inv)
    beta = np.linalg.inv(np.dot(AT_w_inv, A)).dot(AT_w_inv).dot(B)
    bp_rt['error'] = B - A.dot(beta)
    plt.figure(figsize=(5, 5))
    out = bp_rt[w].mean() + 2 * bp_rt[w].std()
    bp_rt[w][bp_rt[w] > out] = out
    plt.scatter(np.sqrt(bp_rt[w]), bp_rt['error'])
    plt.xlabel('sqrt(' + w + ')')
    plt.ylabel('error')
    plt.title('Residual of ' + y[t])


def heter3(x, y, t, f, retn, w='cap', s=1):
    '''
    Returns
    =======
    retn = a + b * f # 用下一期收益率都当期因子进行回归
    检验WLS下的残差是否存在异方差
    返回散点图
    '''
    bp_t0 = x[f][y[t]]
    rt_tn = x[retn][y[t + s]]
    size = x[w][y[t]]
    bp_rt = pd.concat([bp_t0, rt_tn, size], axis=1).dropna()
    X = sm.add_constant(bp_rt[f])
    Y = bp_rt[retn]
    wls = sm.WLS(Y, X, weights=bp_rt[w]).fit()
    beta = wls.params[0]
    bp_rt['error'] = bp_rt[retn] - beta * bp_rt[f] - wls.params[1]
    plt.figure(figsize=(5, 5))
    out = bp_rt[w].mean() + 2 * bp_rt[w].std()
    bp_rt[w][bp_rt[w] > out] = out
    plt.scatter(np.sqrt(bp_rt[w]), bp_rt['error'])
    plt.xlabel('sqrt(' + w + ')')
    plt.ylabel('error')
    plt.title('Residual of ' + y[t])


def f_bar(xz, y, f, period):
    plt.figure(figsize=(10, 5))
    plt.bar(xz, y, label='Return of ' + f)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(y.min() - 0.005, y.max() + 0.005)
    plt.xticks(xz[0:-1:12], period[0:-1:12])
    plt.savefig('Return of ' + f + '.png', bbox_inches='tight')


def f_hist(y, f):
    plt.figure(figsize=(10, 5))
    low = floor(y.min() * 100)
    up = ceil(y.max() * 100)
    bins = pd.Series(range(low, up + 1)) / 100
    plt.hist(y, bins=bins, label='Return of ' + f)
    plt.legend()
    plt.xlim(low / 100, up / 100)
    plt.xticks(bins, bins)
    plt.savefig('Return of ' + f + ' Histgram.png', bbox_inches='tight')


def t_bar(xz, y, f, period):
    plt.figure(figsize=(10, 5))
    plt.bar(xz, y, label='T Value of Return of ' + f)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(y.min() - 1, y.max() + 1)
    plt.xticks(xz[0:-1:12], period[0:-1:12])
    plt.savefig('T Value of ' + f + '.png', bbox_inches='tight')


def ic_bar(xz, y, f, period):
    plt.figure(figsize=(10, 5))
    plt.bar(xz, y, label='IC of ' + f)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(y.min() - 0.01, y.max() + 0.01)
    plt.xticks(xz[0:-1:12], period[0:-1:12])
    plt.savefig('IC of ' + f + '.png', bbox_inches='tight')


def btic_plot(xz, btic, y, f):
    f_bar(xz, btic.beta.values, f, y)
    f_hist(btic.beta, 'SP_TTM')
    t_bar(xz, btic.tvalue.values, f, y)
    ic_bar(xz, btic.ic, f, y)