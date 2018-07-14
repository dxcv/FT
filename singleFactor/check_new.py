# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-12  19:36
# NAME:FT_hp-check_new.py
import multiprocessing

import pandas as pd
import os
import numpy as np

from config import SINGLE_D_INDICATOR, LEAST_CROSS_SAMPLE, SINGLE_D_CHECK
from data.dataApi import read_local
from tools import clean
from scipy import stats, sqrt
import matplotlib.pyplot as plt

G=20

def drawdown(x):
    '''
    Parametes
    =========
    x: DataFrame
        净值数据
    '''
    drawdown = []
    for t in range(len(x)):
        max_t = x[:t + 1].max()
        drawdown_t = min(0, (x[t] - max_t) / max_t)
        drawdown.append(drawdown_t)
    return pd.Series(drawdown).min()

def get_cover_rate(monthly, col):
    monthly['g'] = monthly.groupby('month_end', group_keys=False).apply(
        lambda x: pd.qcut(x['cap'], G,
                          labels=['g{}'.format(i) for i in range(1, G + 1)]))

    cover_rate = monthly.groupby(['month_end', 'g']).apply(
        lambda x: x[col].notnull().sum() / x.shape[0])
    cover_rate = cover_rate.unstack('g') / G
    cover_rate = cover_rate[['g{}'.format(i) for i in range(1, G + 1)]]
    return cover_rate

def get_distribution(comb):
    fdmt = read_local('equity_fundamental_info')
    cap=pd.pivot_table(fdmt,values='cap',index='trd_dt',columns='stkcd').resample('M').last().stack().swaplevel().sort_index()
    grid=pd.concat([comb['g'],cap],axis=1,keys=['g','cap'],join='outer')
    grid.index.names=['stkcd','month_end']
    trd_num=grid.groupby('month_end').apply(lambda x:x['cap'].notnull().sum())
    trd_num.name='trd_num'
    grid=grid.join(trd_num)
    grid['g_cap'] = grid.groupby('month_end', group_keys=False).apply(
        lambda x: pd.qcut(x['cap'], G,
                          labels=['g{}'.format(i) for i in range(1, G + 1)]))

    top=grid[grid['g']=='g{}'.format(G)]
    distribution=top.groupby(['month_end', 'g_cap']).apply(lambda x: x.shape[0] / x['trd_num'].values[0]).unstack().fillna(0)
    return distribution

def get_beta_t_ic(df,col_factor,col_ret):
    '''
    df: DataFrame
        第一列为收益率，第二列为因子值

    计算因子收益率、t值、秩相关系数（IC值）
    '''

    x=df[col_factor]
    y=df[col_ret]
    sl = stats.linregress(x, y)
    beta = sl.slope
    tvalue = sl.slope / sl.stderr
    ic = stats.spearmanr(df)[0]
    return pd.Series([beta,tvalue,ic],index=['beta','tvalue','ic'])

def beta_t_ic_describe(beta_t_ic):
    '''
    beta_t_ic: DataFrame
        含有return, tvalue, ic
    '''
    describe = {
                'Return Mean': beta_t_ic.beta.mean(),
                'months':beta_t_ic.shape[0],
                'Return Std': beta_t_ic.beta.std(),
                'P(t > 0)': len(beta_t_ic[beta_t_ic.tvalue > 0]) / len(beta_t_ic),
                'P(t > 2)': len(beta_t_ic[beta_t_ic.tvalue > 2]) / len(beta_t_ic),
                'P(t <- 2)': len(beta_t_ic[beta_t_ic.tvalue <- 2]) / len(beta_t_ic),
                't Mean': beta_t_ic.tvalue.mean(),
                'rank IC Mean': beta_t_ic.ic.mean(),
                'rank IC Std': beta_t_ic.ic.std(),
                'P(rank IC > 0)': len(beta_t_ic[beta_t_ic.ic > 0]) / len(beta_t_ic.ic),
                'P(rank IC > 0.02)': len(beta_t_ic[beta_t_ic.ic > 0.02]) / len(beta_t_ic.ic),
                'P(rank IC <- 0.02)': len(beta_t_ic[beta_t_ic.ic <- 0.02]) / len(beta_t_ic.ic),
                'rank IC IR': beta_t_ic.ic.mean() / beta_t_ic.ic.std(),
                'tvalue':beta_t_ic.beta.mean()*sqrt(beta_t_ic.shape[0])/beta_t_ic.beta.std()
                }
    describe=pd.Series(describe)
    return describe

def plot_beta_t_ic(beta_t_ic):
    fig = plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(211)
    ax1.bar(beta_t_ic.index, beta_t_ic['beta'], width=20,color='b')
    ax1.set_ylabel('return bar')
    ax1.set_title('factor return')

    ax4=ax1.twinx()
    ax4.plot(beta_t_ic.index,beta_t_ic['beta'].cumsum(),'r-')
    ax4.set_ylabel('cumsum',color='r')
    [tl.set_color('r') for tl in ax4.get_yticklabels()]

    ax2 = plt.subplot(223, sharex=ax1)
    ax2.bar(beta_t_ic.index, beta_t_ic['tvalue'], width=20,color='olive')
    ax2.axhline(y=2, linewidth=1, marker='_', color='r')
    ax2.axhline(y=-2, linewidth=1, marker='_', color='r')
    ax2.set_title('tvalue')

    ax3 = plt.subplot(224, sharex=ax1)
    ax3.bar(beta_t_ic.index, beta_t_ic['ic'], width=20)
    ax3.set_title('ic')
    plt.close()
    return fig

def plot_layer_analysis(g_ret, g_ret_des,distribution,name):
    fig = plt.figure(figsize=(16, 8))
    plt.suptitle(name, fontsize=16)
    ax1 = plt.subplot(221)
    logRet = np.log((1 + g_ret).cumprod())
    logRet['g{}-zz500'.format(G)]=logRet['g{}'.format(G)]-logRet['zz500']
    logRet['g1-zz500'.format(G)]=logRet['g1']-logRet['zz500']

    for col in logRet.columns:
        if col == 'g{}-zz500'.format(G):
            ax1.plot(g_ret.index, logRet[col].values, 'r', label=col, alpha=1,
                     linewidth=1.5)
        elif col == 'g1-zz500'.format(G):
            ax1.plot(g_ret.index, logRet[col].values, 'violet', label=col, alpha=1,
                     linewidth=1.5)
        elif col == 'zz500':
            ax1.plot(g_ret.index, logRet[col].values, 'b', label=col, alpha=1,
                     linewidth=1.5)
        else:
            ax1.plot(g_ret.index, logRet[col].values, alpha=0.8,
                     linewidth=0.5) #hide legend for these portfolios
    ax1.legend()
    ax1.set_title('logRet')
    ax1.grid(True)

    ax2 = plt.subplot(223)
    ax2.bar(g_ret.index, (g_ret['g{}'.format(G)]-g_ret['zz500']).values, width=20, alpha=1,color='b')
    ax2.set_ylabel('return bar', color='b')
    [tl.set_color('b') for tl in ax2.get_yticklabels()]
    ax2.set_title('top minus zz500')

    ax3 = ax2.twinx()
    ax3.plot(g_ret.index, (g_ret['g{}'.format(G)]-g_ret['zz500'] +1).cumprod(), 'r-')
    ax3.set_ylabel('cumprod', color='r')
    [tl.set_color('r') for tl in ax3.get_yticklabels()]

    ax4 = plt.subplot(222)
    barlist=ax4.bar(range(g_ret_des['annual'].shape[0]), g_ret_des['annual'],
                      tick_label=g_ret_des.index, color='b')
    barlist[list(g_ret_des.index).index('zz500'.format(G))].set_color('r')
    ax4.axhline(g_ret_des.loc['zz500','annual'],color='r')
    # ax4.axhline
    ax4.set_title('annual return')

    # distribution
    ax5=plt.subplot(224)
    ax5.stackplot(distribution.index, distribution.T.values, alpha=0.5)
    ax5.set_yticks(np.arange(0,1.2/G,step=(0.2/G)))
    # ax5.set_yticks(np.arange(0,1.2,step=0.2))
    ax5.set_ylim(0,1.0/G)
    ax5.plot(distribution.index,distribution[['g{}'.format(i) for i in range(1,int(G/2)+1)]].sum(axis=1),'b-',label='boundary')
    ax5.set_title('distribution of the top basket')
    plt.close()
    #TODO: add grid
    #TODO： add title
    return fig

def g_ret_describe(g_ret):
    nav = (1 + g_ret).cumprod()
    hpy = nav.iloc[-1, :] - 1
    annual = (nav.iloc[-1, :]) ** (12 / len(g_ret)) - 1
    sigma = g_ret.std() * sqrt(12)
    sharpe = (annual - 0.036) / sigma
    max_drdw = nav.apply(drawdown)

    rela_retn = g_ret.sub(g_ret.iloc[:, -1], axis='index')
    rela_nav = (1 + rela_retn).cumprod()
    rela_annual = (rela_nav.iloc[-1, :]) ** (12 / len(rela_retn)) - 1
    rela_sigma = rela_retn.std() * sqrt(12)
    rela_retn_IR = rela_annual / rela_sigma
    rela_max_drdw = rela_nav.apply(drawdown)

    return pd.DataFrame({
                'hpy': hpy,
                'annual': annual,
                'sigma': sigma,
                'sharpe': sharpe,
                'max_drdw': max_drdw,
                'rela_annual': rela_annual,
                'rela_sigma': rela_sigma,
                'rela_ret_IR': rela_retn_IR,
                'rela_max_drdw': rela_max_drdw})

def get_daily_signal(name):
    df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
    df = df.stack().to_frame().swaplevel().sort_index()
    df.columns = [name]
    fdmt = read_local('equity_fundamental_info')
    data = pd.concat([fdmt, df], axis=1, join='inner')

    data = data.dropna(subset=['type_st', 'young_1year'])
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    data = data.dropna(subset=['wind_indcd', name])
    data = data.groupby('trd_dt').filter(lambda x: x.shape[0] > LEAST_CROSS_SAMPLE) #trick: filter,因为后边要进行行业中性化，太少的样本会出问题

    cleaned_data = clean(data, col=name, by='trd_dt')
    daily = pd.pivot_table(cleaned_data, values=name, index='trd_dt',
                            columns='stkcd').sort_index()
    return daily

def check_factor_monthly(name):
    daily=get_daily_signal(name)
    monthly=daily.resample('M').last()
    monthly.index.name='month_end'
    monthly=monthly.shift(1) #trick: use the signal of the last month

    stk=monthly.stack().swaplevel().sort_index()
    ret_m = read_local('trading_m')['ret_m']
    comb=pd.concat([stk,ret_m],axis=1,keys=[name,'ret_m'],join='inner').dropna().sort_index()
    comb.index.names=['stkcd','month_end']
    beta_t_ic = comb.groupby('month_end').apply(get_beta_t_ic, name, 'ret_m')
    # layer test
    comb['g'] = comb.groupby('month_end', group_keys=False).apply(
        lambda x: pd.qcut(x[name], G,
                          labels=['g{}'.format(i) for i in range(1, G + 1)]))

    g_ret = comb.groupby(['month_end', 'g'])['ret_m'].mean().unstack('g')
    g_ret.columns = g_ret.columns.tolist()
    g_ret['zz500']=read_local('indice_m')['zz500_ret_m']
    g_ret=g_ret.dropna()

    distribution=get_distribution(comb)
    beta_t_ic_des = beta_t_ic_describe(beta_t_ic)
    fig_beta_t_ic = plot_beta_t_ic(beta_t_ic)

    g_ret_des = g_ret_describe(g_ret)
    fig_g = plot_layer_analysis(g_ret, g_ret_des,distribution,name)

    dfs={'beta_t_ic':beta_t_ic,
       'beta_t_ic_des':beta_t_ic_des,
       'g_ret':g_ret,
       'g_ret_des':g_ret_des
       }
    figs={'fig_beta_t_ic':fig_beta_t_ic,
          'fig_g':fig_g}
    return dfs,figs

def save_results(dfs,figs,name):
    directory=os.path.join(SINGLE_D_CHECK,name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for k in dfs.keys():
        dfs[k].to_csv(os.path.join(directory, k + '.csv'))

    for k in figs.keys():
        figs[k].savefig(os.path.join(directory, k + '.png'))

def check_with_name(name):
    try:
        dfs,figs=check_factor_monthly(name)
        save_results(dfs,figs,name)
        print(name)
    except:
        print('{}----------> wrong'.format(name))

def debug():
    name='Q__roe'
    dfs, figs = check_factor_monthly(name)
    save_results(dfs, figs, name)

def main():
    fns=os.listdir(SINGLE_D_INDICATOR)
    fns=[fn for fn in fns if fn.endswith('.pkl')]
    names=[fn[:-4] for fn in fns]
    print(len(names))
    checked=os.listdir(SINGLE_D_CHECK)
    names=[n for n in names if n not in checked]
    print(len(names))
    pool = multiprocessing.Pool(4)
    pool.map(check_with_name, names)

# if __name__ == '__main__':
#     main()

