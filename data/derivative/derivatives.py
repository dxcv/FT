# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-15  13:54
# NAME:FT-derivatives.py
import pandas as pd

import os
from config import D_DRV
from data.dataApi import read_local, get_dataspace
from config import D_FILESYNC_ADJ


def _ttm(s):
    df=s.reset_index()
    del df['stkcd']
    df=df.set_index('report_period')
    df.columns=['raw']
    df['year_before']=df['raw'].shift(4)
    v=df.loc[df.index-pd.offsets.YearEnd(1)]['raw'].values
    df['last_year_end']=v
    df['ttm']=df['raw']-df['year_before']+df['last_year_end']
    return df['ttm']

def test_ttm():
    balance = read_local('equity_selected_cashflow_sheet')
    col = 'depr_fa_coga_dpba'
    ttm=balance[col].groupby('stkcd').apply(_ttm)

def cal_ebitda():
    '''
    有大量数据缺失df.notnull().sum()
    Returns:

    '''
    #EBITDA=息税前利润+当期计提折旧与摊销  wind code generator EBITDA(反推法)
    fields=['net_profit_incl_min_int_inc','inc_tax','fin_exp',
            'depr_fa_coga_dpba','amort_intang_assets','amort_lt_deferred_exp']
    df=get_dataspace(fields)

    # 如果inc_tax 和 fin_exp 中有一项有数据，就保留该item并把缺失值用0填充
    tax_exp = df[['inc_tax', 'fin_exp']]
    tax_exp = tax_exp.dropna(how='all')
    tax_exp = tax_exp.fillna(0)
    tax_exp_sum = tax_exp.sum(axis=1).reindex(df.index)
    df['ebit']=df['net_profit_incl_min_int_inc']+tax_exp_sum

    # 如果 三个指标中有一个存在，就保留，并用0填充其他的缺失值
    amorts=df[['depr_fa_coga_dpba','amort_intang_assets','amort_lt_deferred_exp']]
    amorts=amorts.dropna(how='all').fillna(0)
    amorts_sum=amorts.sum(axis=1).reindex(df.index)

    df['ebitda']=df['ebit']+amorts_sum

    df[['trd_dt','ebit','ebitda']].to_pickle(os.path.join(D_DRV, 'ebit.pkl'))

def cal_netAsset():
    name='netAsset'
    col1='tot_assets'
    col2='tot_liab'
    df=get_dataspace([col1,col2])
    df[name]=df[col1]-df[col2]
    df[['trd_dt',name]].to_pickle(os.path.join(D_DRV,'{}.pkl'.format(name)))

def cal_netNonOI():
    name='netNonOI'
    col1='non_oper_rev'
    col2='non_oper_exp'
    df=get_dataspace([col1,col2])
    df[name]=df[col1]-df[col2]
    df[['trd_dt',name]].to_pickle(os.path.join(D_DRV,'{}.pkl'.format(name)))

def cal_periodCost():
    name='periodCost'
    col1='oper_cost'
    col2='gerl_admin_exp'
    col3='fin_exp'
    df=get_dataspace([col1,col2,col3])
    df[name]=df[col1]+df[col2]+df[col3]
    df[['trd_dt', name]].to_pickle(os.path.join(D_DRV, '{}.pkl'.format(name)))

def cal_receivable():
    name='receivable'
    col1='notes_rcv'
    col2='acct_rcv'
    df=get_dataspace([col1,col2])
    df[name]=df[col1]+df[col2]
    df[['trd_dt', name]].to_pickle(os.path.join(D_DRV, '{}.pkl'.format(name)))

def cal_payable():
    name='payable'
    col1='notes_payable'
    col2='acct_payable'
    df=get_dataspace([col1,col2])
    df[name]=df[col1]+df[col2]
    df[['trd_dt', name]].to_pickle(os.path.join(D_DRV, '{}.pkl'.format(name)))

def cal_grossIncome():
    name='grossIncome'
    col1='oper_rev'
    col2='oper_cost'
    df = get_dataspace([col1, col2])
    df[name] = df[col1] - df[col2]
    df[['trd_dt', name]].to_pickle(os.path.join(D_DRV, '{}.pkl'.format(name)))

# cal_netAsset()
# cal_netNonOI()
# cal_periodCost()
# cal_receivable()
# cal_payable()
# cal_grossIncome()



def daily2monthly(x):
    ohlc_dict={
                # 'preclose':'last',
               # 'open':'first',
               # 'high':'max',
               # 'low':'min',
               # 'close':'last',
               'volume':'sum',
               # 'amount':'sum',
               # 'adjpreclose':'last',
               'adjopen':'first',
               'adjhigh':'max',
               'adjlow':'min',
               'adjclose':'last',
               # 'avgprice':'mean',
               'trd_dt':'last'
               }
    monthly=x.resample('M',on='trd_dt',closed='right',label='right').apply(ohlc_dict)
    # monthly=x.resample('M',how=ohlc_dict,closed='right',label='right',on='trd_dt')
    monthly['ret_m']=monthly['adjclose'].pct_change()
    '''
    asfreq('M') 这一行可以省略，因为daily的数据没有缺失交易日
    '''
    monthly=monthly.asfreq('M') #span
    return monthly

def get_monthly_trading_data():
    daily=read_local('equity_selected_trading_data')
    daily=daily.reset_index()
    monthly=daily.groupby('stkcd').apply(daily2monthly)
    monthly=monthly.rename(columns={'adjopen':'open',
                            'adjhigh':'high',
                            'adjlow':'low',
                            'adjclose':'close',
                            })
    monthly=monthly[['trd_dt']+[col for col in monthly.columns if col!='trd_dt']]
    monthly.index.names=['stkcd','month_end']

    monthly.to_pickle(os.path.join(D_FILESYNC_ADJ,'trading_m.pkl'))

# 原始数据中在停牌的时候股价是向前填充，ohlc 均为上一个交易日的收盘价，交易量为0

#TODO: use this method to handle quarterly data

# get_monthly_trading_data()

def get_monthly_cap():
    daily=read_local('equity_fundamental_info')
    daily=daily.reset_index()
    monthly=daily[['stkcd','trd_dt','tot_shr','float_a_shr','freeshares','cap',
                   'freefloat_cap']].groupby('stkcd').apply(
        lambda x:x.resample('M',on='trd_dt',closed='right',label='right').last())

    monthly.to_pickle(os.path.join(D_FILESYNC_ADJ,'shr_and_cap.pkl'))

