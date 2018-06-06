# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:56:34 2018

@author: XQZhu
"""
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from scipy import stats

def layer_result(x, y, mkt_idx, mkt_name, rf, n=10, w='equal'):
    '''
    Parameters
    ==========
    x:
        包含股票代码、市值、分层标记、收益率的DataFrame
    y:
        period
    mkt_idx:
        市场指数的月度收益率
    mkt_name:
        市场指数的名称
    rf:
        无风险利率
    n:
        层数
    w:
        计算组合收益率的权重，equal表示等权重，cap表示市值加权
    
    Returns
    =======
    return lt_result, accum # 以元组的形式返回
    lt_result:
        分层测试的评价指标
    accum:
        净值序列
    '''
    def layertest(t):
        x.loc[:, 'equal'] = np.ones(len(x))
        # 组合的股票构成和权重在当期确定
        port_t0 = x.loc[y[t], ['stkcd', 'cap', 'equal', 'layers']]
        retn_t1 = x.loc[y[t + 1],['stkcd', 'retn']] # 收益率为下一期收益率
        port = pd.merge(port_t0, retn_t1, on='stkcd').dropna()
        port_re = port.groupby('layers').apply(lambda x: np.average(x.retn, weights=x[w]))
        return port_re

    def drawdown(x):
        drawdown = []
        for t in range(len(x)):
            max_t = x[:t + 1].max()
            drawdown_t = min(0, (x[t] - max_t) / max_t)
            drawdown.append(drawdown_t)
        return pd.Series(drawdown).min()
    
    month = pd.DataFrame({y[t]: layertest(t) for t in range(len(y) -1)}).T
    month['l_s'] = month.iloc[:, n-1] - month.iloc[:, 0] # 多空组合收益率
    month = pd.merge(month, mkt_idx, left_index=True, right_index=True, how='left')
    
    # 组合超额收益等于组合收益率减去（beta * 基准收益率）
#    alpha = []
    res = pd.DataFrame()
    cols = month.columns.tolist()
    for i in range(n + 2):
        sl = stats.linregress(month.iloc[:, -1], month.iloc[:, i])
#        alpha.append(sl.intercept)
        res[cols[i]] = month.iloc[:, i] - sl.slope * month.iloc[:, -1]
    rlt_ann = res.mean() * 12
    rlt_vol = res.std() * sqrt(12) # 年化波动率
    ir =  rlt_ann / rlt_vol
    
    accum = (1 + month).cumprod()
    hpy = accum.iloc[-1, :] - 1
    annual = (accum.iloc[-1, :]) ** (12 / (len(y) -1)) - 1
    sigma = np.std(month) * sqrt(12) # 年华波动率
    sharpe = (annual - rf / 100) / sigma
    max_drawdown = accum.apply(drawdown)
    
    # 组合超额收益等于组合收益率减去基准收益率
#    rlt_retn = month.iloc[:, :n + 2].sub(month[mkt_name ], axis='index')
#    rlt_ann = rlt_retn.mean() * 12 # 年化算数平均收益率
#    rlt_vol = rlt_retn.std() * sqrt(12) # 年化波动率
#    ir =  rlt_ann / rlt_vol
    
    # 组合的额超额收益率等于组合净值除以基准净值的百分比
    rlt2_nav = accum.iloc[:, :n + 2].div(accum[mkt_name], axis='index')
    rlt2_retn = rlt2_nav.apply(lambda x: x.pct_change()).dropna()
    # 年化几何平均收益率
    rlt2_ann = rlt2_nav.iloc[-1, :] ** (12 / len(rlt2_retn)) - 1
    rlt2_vol = np.std(rlt2_retn) * sqrt(12) # 年化波动率
    ir2 = np.mean(rlt2_retn) * 12 / rlt2_vol
    rlt2_maxdrdw = rlt2_nav.apply(drawdown)    


    result = {'Accum Return': hpy, 
              'Ann Return': annual, 
              'Vol': sigma,
              'Sharpe': sharpe, 
              'Max Drdw': max_drawdown,
              'R Ann Return': rlt_ann,
              'R Vol': rlt_vol,
              'IR': ir,
              'R2 Ann Return': rlt2_ann,
              'R2 Vol': rlt2_vol,
              'IR2': ir2,
              'R Max Drdw': rlt2_maxdrdw}
    result = pd.DataFrame(result, 
                          columns=['Accum Return', 'Ann Return', 'Vol', 'Sharpe', 
                                   'Max Drdw', 'R Ann Return', 'R Vol', 'IR', 
                                   'R2 Ann Return', 'R2 Vol', 'IR2', 'R Max Drdw'])
    return result.T, accum, month

def nav_plot(xz, x, y, f, mkt_name, n=10, w='equal'):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.plot(xz, x.iloc[:, i], label=x.columns[i])
    plt.plot(xz, x['l_s'], 'r--', label='l_s')
    plt.plot(xz, x[mkt_name], 'c--x', label=mkt_name)
    plt.legend()
    plt.title('NAV of ' + f +' Portfolios (' + w.capitalize() +'-Weighted)')
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(0, x.max().max() + 1)
    plt.xticks(xz[0:-1:12], y[0:-1:12])
    plt.savefig('NAV of ' + f +' LayerTest (' + w[0].upper() +'W).png', bbox_inches='tight')

def ann_bar(x, f, w='equal', n=10):
    plt.figure(figsize=(10, 5))
    xz2 = list(range(n + 2))
    plt.bar(xz2, x, label='Annualized Return')
    plt.legend()
    plt.title('Monotonicity of Return of ' + f +' Portfolios (' + w.capitalize() +'-Weighted)')
    plt.xlim(- 1, n + 2)
    plt.ylim(x.min() - 0.01, x.max() + 0.01)
    plt.xticks(xz2, x.index.tolist())
    plt.savefig('Ann of ' + f + ' Portfolios (' + w[0].upper() +'W).png', bbox_inches='tight')