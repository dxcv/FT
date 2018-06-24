# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-23  21:11
# NAME:FT-new_check_factors_0623.py
from math import floor, ceil, sqrt

from data.dataApi import read_local
import pandas as pd
import os
from config import SINGLE_D_INDICATOR, SINGLE_D_CHECK
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

G = 10


def change_index(df):
    df = df.reset_index().sort_values(['stkcd', 'trd_dt', 'report_period'])
    # 如果在相同的trd_dt有不同的report_period记录，取report_period较大的那条记录
    df = df[~df.duplicated(['stkcd', 'trd_dt'], keep='last')]
    del df['report_period']
    df=df.set_index(['stkcd','trd_dt']).dropna()
    return df


def outlier(x, k=4.5):
    '''
    Parameters
    ==========
    x:
        原始因子值
    k = 3 * (1 / stats.norm.isf(0.75))
    '''
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    uplimit = med + k * mad
    lwlimit = med - k * mad
    y = np.where(x >= uplimit, uplimit, np.where(x <= lwlimit, lwlimit, x))
    return pd.DataFrame(y, index=x.index)

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
    beta = np.linalg.lstsq(A, y,rcond=None)[0]
    res = y - np.dot(A, beta)
    return res


def clean(df, col):
    '''
    Parameters
    ==========
    df: DataFrame
        含有因子原始值、市值、行业代码
    col:
        因子名称
    '''
    df[col + '_out']=df.groupby('month_end')[col].apply(outlier)
    df[col + '_zsc']=df.groupby('month_end')[col + '_out'].apply(z_score)
    df['wind_2'] = df['wind_indcd'].apply(str).str.slice(0, 6) # wind 2 级行业代码
    df = df.join(pd.get_dummies(df['wind_2'], drop_first=True))
    df['ln_cap'] = np.log(df['cap'])
    industry = list(np.sort(df['wind_2'].unique()))[1:]
    df[col + '_neu'] = df.groupby('month_end', group_keys=False).apply(neutralize, col + '_zsc', industry)

    del df[col]
    del df[col + '_out']
    del df[col + '_zsc']
    df=df.rename(columns={col + '_neu':col})
    return df

def get_beta_t_ic(df,col_factor,col_ret):
    '''
    df: DataFrame
        第一列为收益率，第二列为因子值

    计算因子收益率、t值、秩相关系数（IC值）
    '''

    x=df[col_factor]
    y=df[col_ret]
    sl = stats.linregress(x, y) #review: intercept ?
    beta = sl.slope
    tvalue = sl.slope / sl.stderr
    ic = stats.spearmanr(df)[0]
    return pd.Series([beta,tvalue,ic],index=['beta','tvalue','ic'])


def beta_t_ic_describe(beta_t_ic):
    '''
    beta_t_ic: DataFrame
        含有return, tvalue, ic
    '''
    describe = {'Return Mean': beta_t_ic.beta.mean(),
              'Return Std': beta_t_ic.beta.std(),
              'P(t > 0)': len(beta_t_ic[beta_t_ic.tvalue > 0]) / len(beta_t_ic),
              'P(|t| > 2)': len(beta_t_ic[abs(beta_t_ic.tvalue) > 2]) / len(beta_t_ic),
              '|t| Mean': abs(beta_t_ic.tvalue).mean(),
              'IC Mean': beta_t_ic.ic.mean(),
              'IC Std': beta_t_ic.ic.std(),
              'P(IC > 0)': len(beta_t_ic[beta_t_ic.ic > 0]) / len(beta_t_ic.ic),
              'P(IC > 0.02)': len(beta_t_ic[beta_t_ic.ic > 0.02]) / len(beta_t_ic.ic),
              'IC IR': beta_t_ic.ic.mean() / beta_t_ic.ic.std()}
    describe=pd.Series(describe)
    return describe


def plot_beta_t_ic(beta_t_ic):
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(311)
    ax1.bar(beta_t_ic.index, beta_t_ic['beta'], width=20)
    ax1.set_title('beta')

    ax2 = plt.subplot(312, sharex=ax1)
    ax2.bar(beta_t_ic.index, beta_t_ic['tvalue'], width=20)
    ax2.axhline(y=2, linewidth=0.5, marker='_', color='r')
    ax2.axhline(y=-2, linewidth=0.5, marker='_', color='r')
    ax2.set_title('tvalue')

    ax3 = plt.subplot(313, sharex=ax1)
    ax3.bar(beta_t_ic.index, beta_t_ic['ic'], width=20)
    ax3.set_title('ic')
    return fig


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


def plot_g_ret(g_ret):
    cumprod=(1 + g_ret).cumprod()
    xz = list(range(len(cumprod)))
    xn = cumprod.index.strftime('%Y-%m')
    fig_layer = plt.figure(figsize=(10, 5))

    for col in cumprod.columns:
        if col=='g{}_g1'.format(G):
            plt.plot(xz, cumprod[col], 'r', label=col, alpha=1, linewidth=1)
        elif col=='zz500':
            plt.plot(xz, cumprod[col], 'b', label=col, alpha=1, linewidth=1)
        else:
            plt.plot(xz, cumprod[col], label=col, alpha=0.8, linewidth=0.5)
    plt.legend()
    plt.title('cumprod')
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(0, cumprod.max().max() + 1)
    plt.xticks(xz[0:-1:12], xn[0:-1:12])
    return fig_layer


def plot_annual_bar(g_ret_des):
    annual=g_ret_des['annual']
    fig = plt.figure(figsize=(10, 5))
    xz = list(range(G + 2))
    plt.bar(xz,annual, label='Annualized Return')
    plt.legend()
    plt.xlim(- 1, G + 2)
    plt.ylim(annual.min() - 0.01, annual.max() + 0.01)
    xn = annual.index.tolist()
    plt.xticks(xz, xn)
    return fig


def plot_tmb(g_ret):
    fig_tmb, ax1 = plt.subplots(figsize=(16, 8))
    ax1.bar(g_ret.index, g_ret['g10_g1'].values, width=20, alpha=1)
    ax1.set_xlabel('month_end')
    ax1.set_ylabel('return bar', color='b')
    [tl.set_color('b') for tl in ax1.get_yticklabels()]

    ax2 = ax1.twinx()
    ax2.plot(g_ret.index, (1 + g_ret['g10_g1']).cumprod(), 'r-')
    ax2.set_ylabel('cumprod', color='r')
    [tl.set_color('r') for tl in ax2.get_yticklabels()]
    return fig_tmb

def get_result(df,ret_1m,fdmt,zz500_ret_1m):
    col=df.columns[0]

    # merge and filter sample
    data=pd.merge(fdmt.reset_index(),df.reset_index(),on=['stkcd','trd_dt'],how='left')
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    data = data.groupby('stkcd').ffill(limit=400) # review: 向前填充最最多400个交易日

    #resample monthly to ease the calculation,but remember that resample will convert the trading date to calendar date(month end).
    monthly=data.groupby('stkcd').resample('M',on='trd_dt').last().dropna()
    del monthly['trd_dt']#trick: trd_dt is no longer useful,we should get it
    monthly.index.names=['stkcd','month_end']
    monthly=monthly.groupby('month_end').filter(lambda x:x.shape[0]>300) #trick: filter,因为后边要进行行业中性化，太少的样本会出问题
    monthly=clean(monthly, col) # outlier,z_score,nutrualize

    #cross sectional analyse and layers test
    comb=pd.concat([ret_1m,monthly[col]],axis=1,join='inner').dropna()
    comb=comb.groupby('month_end').filter(lambda x:x.shape[0]>=50)#Trick:后边要进行分组分析，数据样本太少的话没法做
    if comb.shape[0]>0:
        # cross section    beta tvalue ic
        beta_t_ic=comb.groupby('month_end').apply(get_beta_t_ic,col,'ret_1m')

        #layer test
        comb['g']=comb.groupby('month_end',group_keys=False).apply(
            lambda x:pd.qcut(x[col],G,labels=['g{}'.format(i) for i in range(1,G+1)]))
        g_ret=comb.groupby(['month_end','g'])['ret_1m'].mean().unstack('g')
        g_ret.columns=g_ret.columns.tolist()
        g_ret['g{}_g1'.format(G)]=g_ret['g{}'.format(G)]-g_ret['g1']#top minus bottom
        g_ret['zz500']=zz500_ret_1m
        g_ret=g_ret.dropna()
        return beta_t_ic,g_ret





# fn='Q__mlev.pkl'
#
# fdmt = read_local('equity_fundamental_info')[
#     ['cap', 'type_st', 'wind_indcd', 'young_1year']]
#
# ret_1m = read_local('trading_m')['ret_1m']  # review : trading date
# ret_1m = ret_1m.reset_index().groupby('stkcd').resample(
#     'M',on='trd_dt').last().dropna()[['ret_1m']]
# ret_1m.index.names = ['stkcd', 'month_end']
#
# zz500_ret_1m=read_local('indice_m')['zz500_ret_1m'] #review:trading date
# zz500_ret_1m=zz500_ret_1m.resample('M').last()
# zz500_ret_1m.index.name='month_end'
#
#
# df=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR,fn))
# df=change_index(df)
# # df=df[:int(df.shape[0]/20)]#review
# beta_t_ic,g_ret=get_result(df,ret_1m,fdmt,zz500_ret_1m)
#
# col=df.columns[0]
# directory=os.path.join(SINGLE_D_CHECK,col)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# beta_t_ic.to_csv(os.path.join(directory,'beta_t_ic.csv'))
# g_ret.to_csv(os.path.join(directory,'g_ret.csv'))




#-------------------------present results--------------------------------------
directory=r'D:\zht\database\quantDb\internship\FT\singleFactor\check\Q__mlev'
beta_t_ic=pd.read_csv(os.path.join(directory,'beta_t_ic.csv'),index_col=0,encoding='utf8',parse_dates=True)
g_ret=pd.read_csv(os.path.join(directory,'g_ret.csv'),index_col=0,encoding='utf8',parse_dates=True)

des=beta_t_ic_describe(beta_t_ic)
fig_beta_t_ic=plot_beta_t_ic(beta_t_ic)

g_ret_des=g_ret_describe(g_ret)
fig_layer=plot_g_ret(g_ret)
fig_annual_bar=plot_annual_bar(g_ret_des)
fig_tmb=plot_tmb(g_ret)


fig_beta_t_ic.show()
fig_annual_bar.show()
fig_layer.show()
fig_tmb.show()


fig = plt.figure(figsize=(16, 8))
ax1 = plt.subplot(221)
# ax1=plt.subplot2grid((2,2),(0,0),colspan=1,rowspan=1)
cumprod = (1 + g_ret).cumprod()
for col in cumprod.columns:
    if col == 'g{}_g1'.format(G):
        ax1.plot(g_ret.index, cumprod[col], 'r', label=col, alpha=1, linewidth=1)
    elif col == 'zz500':
        ax1.plot(g_ret.index, cumprod[col], 'b', label=col, alpha=1, linewidth=1)
    else:
        ax1.plot(g_ret.index, cumprod[col], label=col, alpha=0.8, linewidth=0.5)
ax1.set_title('cumprod')

ax2 = plt.subplot(223)
# ax3=plt.subplot2grid((2,2),(1,0),colspan=1,rowspan=1)
ax2.bar(g_ret.index, g_ret['g10_g1'].values, width=20,alpha=1)
ax2.set_xlabel('month_end')
ax2.set_ylabel('return bar', color='b')
[tl.set_color('b') for tl in ax2.get_yticklabels()]

ax3 = ax2.twinx()
ax3.plot(g_ret.index, cumprod['g{}_g1'.format(G)], 'r-')
ax3.set_ylabel('cumprod', color='r')
[tl.set_color('r') for tl in ax3.get_yticklabels()]


ax4 = plt.subplot(122)
# ax2=plt.subplot2grid((2,2),(1,1),colspan=1,rowspan=2)
colors
ax4.bar(range(g_ret_des['annual'].shape[0]),g_ret_des['annual'],tick_label=g_ret_des.index)
ax4.set_title('annual return')



fig.show()








