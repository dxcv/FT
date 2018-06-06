# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:51:00 2018

@author: XQZhu
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
import statsmodels.api as sm

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
    tvalue = sl.slope / sl.stderr
    ic_value = stats.spearmanr(bp_rt)[0]
    btic = pd.DataFrame({'beta': [beta], 'tvalue': [tvalue], 'ic': [ic_value]},
                         columns=['beta', 'tvalue', 'ic'])
    return btic

def premium2_ic(x, y, f, retn, t, s, w='cap'): # numpy.linalg求解WLS
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
    w_inv = np.diag(np.sqrt(bp_rt[w])) #市值平方根作为权重
    AT_w_inv = np.dot(A.T, w_inv)
    beta = np.linalg.inv(np.dot(AT_w_inv, A)).dot(AT_w_inv).dot(B)
    sse = np.sum((B - A.dot(beta)) ** 2)
    tvalue = beta[0] * sqrt(len(a)) * np.std(a) / sqrt(sse / (len(a) - 2))
    ic_value = stats.spearmanr(bp_rt.loc[:, [f, retn]])[0]
    btic = pd.DataFrame({'beta': [beta[0]], 'tvalue': [tvalue], 'ic': [ic_value]}, 
                         columns=['beta', 'tvalue', 'ic'])
    return btic

def premium3_ic(x, y, f, retn, t, s, w='cap'): # sm.WLS求解
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
            premium_result,\
            columns=['Factor return mean ', 'Factor return std','P(t > 0)', 
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
    bp_rt['error'] = bp_rt[retn] -  sl.intercept - beta * bp_rt[f]
    
#    plt.figure(figsize=(5, 5))
#    plt.scatter(bp_rt[f] ** 2, bp_rt['error'])
#    plt.xlabel(f)
#    plt.ylabel('error')
#    plt.title('Residual of ' + y[t])
    
    plt.figure(figsize=(5, 5))
    out = bp_rt[w].mean() + 2 * bp_rt[w].std()
    bp_rt[w][bp_rt[w] > out] = out
    plt.scatter(np.sqrt(bp_rt[w]), bp_rt['error'])
    plt.xlabel('sqrt('+ w +')')
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
    plt.xlabel('sqrt('+ w +')')
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
    plt.xlabel('sqrt('+ w +')')
    plt.ylabel('error')
    plt.title('Residual of ' + y[t])
    
def f_bar(xz, y, f, period):
    plt.figure(figsize=(10, 5))
    plt.bar(xz, y, label='Return of ' + f)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(y.min() - 0.005, y.max() + 0.005)
    plt.xticks(xz[0:-1:12], period[0:-1:12])
    plt.savefig('Return of ' + f +'.png', bbox_inches='tight')
    
def f_hist(y, f):
    plt.figure(figsize=(10, 5))
    low = floor(y.min() * 100)
    up = ceil(y.max() * 100)
    bins=pd.Series(range(low, up + 1)) / 100
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