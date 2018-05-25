# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-24  14:55
# NAME:FT-calculate_factors.py
from data.dataApi import read_local
from data.get_base import read_base
import data.database_api.database_api as dbi
from singleFactor.factors.test_single_factor import test
import pandas as pd
import matplotlib.pyplot as plt

from tools import handle_duplicates


def growth_yoy_df(df, col):
    # 同比增长比率
    #TODO: how about missing data? we should use stkcd and report date as primary keys
    #TODO: sort
    name='{}_yoy'.format(col)
    df[name]=df[[col]].groupby('stkcd').apply(
        lambda x:x.pct_change(periods=4,limit=4))
    test(df[[name]],name)

def growth_yoy_tbname(tbname, col):
    # 同比增长率
    df=dbi.get_stocks_data(tbname, [col])
    growth_yoy_df(df, col)

def get_saleEarnings_sq_yoy():
    #单季度营业利润同比增长率
    tbname='equity_selected_income_sheet_q'
    col='oper_profit'
    growth_yoy_tbname(tbname, col)

def get_earnings_sq_yoy():
    #单季度净利润同比增长率
    tbname='equity_selected_income_sheet_q'
    col='net_profit_excl_min_int_inc'
    growth_yoy_tbname(tbname, col)

def get_sales_sq_yoy():
    #单季度营业收入同比增长率
    tbname='equity_selected_income_sheet_q'
    col='oper_rev'
    growth_yoy_tbname(tbname, col)

def get_eps1Ygrowth_yoy():
    #每股收益同比增长率
    cap_stk=dbi.get_stocks_data('equity_selected_balance_sheet',['cap_stk'])
    income=dbi.get_stocks_data('equity_selected_income_sheet',['net_profit_excl_min_int_inc'])
    cap_stk=handle_duplicates(cap_stk)
    income=handle_duplicates(income)
    df=pd.concat([cap_stk,income],axis=1)
    df['income_per_share']=df['cap_stk']/df['net_profit_incl_min_int_inc']
    growth_yoy_df(df, 'income_per_share')

def get_ocfGrowth_yoy():
    #经营现金流增长率
    tbname='equity_selected_cashflow_sheet'
    name='net_cash_flows_oper_act'
    growth_yoy_tbname(tbname,name)

def test_ltg1(tbname,col,name):
    df=dbi.get_stocks_data(tbname,[col])
    df[name]=df[[col]].groupby('stkcd').apply(
        lambda x:x.rolling(20).apply(lambda s:(s[-1]-s[0])/s[0]))

    #TODO:
    test(df[[name]],name)

def test_ltg(tbname,col,name):
    df=read_local(tbname,col)

    def cal_ltg(x,col):
        x=x.reset_index()
        x=x.set_index('report_period')
        qrange=pd.date_range(x.index.min(),x.index.max(),freq='Q')
        x=x.reindex(qrange)
        x['result']=x[col].rolling(20).apply(lambda s:(s[-1]-s[0])/s[0])
        x=x.set_index('trd_dt').dropna()
        x=x.sort_index()
        x=handle_duplicates(x)
        return x['result']
    df[name]=df.groupby('stkcd').apply(cal_ltg,col)
    test(df[[name]],name)

def get_earnings_ltg():
    #净利润过去 5 年历史增长率
    tbname='equity_selected_income_sheet'
    col='net_profit_excl_min_int_inc'
    name='earnings_ltg'
    test_ltg(tbname,col,name)

def get_sales_ltg():
    # 营业收入过去 5 年历史增长率
    tbname = 'equity_selected_income_sheet'
    col = 'oper_rev'
    name='sales_ltg'

    #TODO: data loss is netative?
    test_ltg1(tbname,col,name)

# get_sales_ltg()



#净利润未来 1 年预期增长率
def get_earnings_sfg():
    name='earnings_sfg'
    predict=dbi.get_stocks_data('equity_consensus_forecast',
                           ['benchmark_yr','est_net_profit_FTTM'])
    predict['benchmark_yr']=pd.to_datetime(predict['benchmark_yr'])
    predict=predict.reset_index()

    # est=predict.groupby(['stkcd','benchmark_yr']).apply(
    #     lambda x:pd.Series([x['est_net_profit_FTTM'].mean(),x['trd_dt'].iloc[-1]],
    #                        index=['est_mean','trd_dt'])).reset_index()
    est=predict.groupby(['stkcd','benchmark_yr']).first().reset_index() #取第一家预测值
    est=est.rename(columns={'benchmark_yr':'key_date'})


    real=dbi.get_stocks_data('equity_selected_income_sheet',
                             ['report_period','net_profit_excl_min_int_inc'])
    real['report_period']=pd.to_datetime(real['report_period'])
    real=real.reset_index().rename(columns={'report_period':'key_date'})
    del real['trd_dt']

    df=pd.merge(est,real,on=['stkcd','key_date'])
    df[name]=(df['est_net_profit_FTTM']-df['net_profit_excl_min_int_inc']
              )/df['net_profit_excl_min_int_inc']
    # df[name]=(df['est_mean']-df['net_profit_excl_min_int_inc']
    #           )/df['net_profit_excl_min_int_inc']

    df=df.set_index(['trd_dt','stkcd'])
    test(df[[name]],name)


def get_g_netcashflow():
    #净现金流增长率
    tbname = 'equity_selected_cashflow_sheet'
    col = 'net_cash_flows_oper_act'
    growth_yoy_tbname(tbname,col)

def get_g_netProfit12Qavg():
    #过去 12 个季度净利润平均年增长率
    tbname = 'equity_selected_income_sheet_q'
    col = 'net_profit_excl_min_int_inc'
    name='g_netProfit12Qavg'
    df = dbi.get_stocks_data(tbname, [col])
    df['yoy']=df.groupby('stkcd').apply(lambda df:df.pct_change(periods=4))
    df[name]=df[['yoy']].groupby('stkcd').apply(
        lambda x:x.rolling(12,min_periods=12).mean())
    test(df[[name]],name)

def get_g_totalOperatingRevenue12Qavg():
    #过去 12 个季度营业总收入平均年增长率
    tbname='equity_selected_income_sheet_q'
    col='tot_oper_rev'
    name='g_totalOperatingRevenue12Qavg'
    df = dbi.get_stocks_data(tbname, [col])
    df['yoy'] = df.groupby('stkcd').apply(lambda df: df.pct_change(periods=4))
    df[name] = df[['yoy']].groupby('stkcd').apply(
        lambda x: x.rolling(12, min_periods=12).mean())
    test(df[[name]], name)

def get_g_totalAssets():
    #总资产增长率
    tbname='equity_selected_balance_sheet'
    col='tot_assets'
    growth_yoy_tbname(tbname,col)

#净利润增量连续大于 0 的期数
def get_g_earningsCHG():
    raise NotImplementedError

    tbname = 'equity_selected_income_sheet'
    col = 'net_profit_excl_min_int_inc'
    name='g_earningsCHG'
    df=dbi.get_stocks_data(tbname,[col])
    df['yoy']=df[[col]].groupby('stkcd').apply(
        lambda x:x.pct_change(periods=4)
    )

    def func(x):
        print(x.index[0])
        counts=[]
        for ind in x.index:
            sub=x.loc[:ind]
            i=0
            mark=True
            while mark:
               if sub['yoy'].values[-(i+1)]>0:
                   i+=1
               else:
                   mark=False
            counts.append(i)
        return pd.Series(counts,index=x.index.get_level_values('trd_dt'))

    df[name]=df.groupby('stkcd').apply(func)

def get_saleEarnings_sq_yoy_5():
    #EPS5 年复合增长率
    tbname='equity_selected_income_sheet'
    col='oper_profit'
    name='g_epscagr5'
    df=dbi.get_stocks_data(tbname,[col])
    df[name]=df[[col]].groupby('stkcd').apply(
        lambda x:x.pct_change(periods=20)
    ) #直接求20个季度的增长率等价于5年复合增长率
    test(df[[name]],name)


def get_g_netOperateCashFlowPerShare():
    #每股经营活动净现金流增长率
    cash_flow=dbi.get_stocks_data('equity_selected_cashflow_sheet',
                                  ['net_cash_flows_oper_act'])
    cap_stk=dbi.get_stocks_data('equity_selected_balance_sheet',
                                ['cap_stk'])
    cash_flow=handle_duplicates(cash_flow)
    cap_stk=handle_duplicates(cap_stk)
    df=pd.concat([cash_flow,cap_stk],axis=1)
    df['cash_flow_per_share']=df['net_cash_flows_oper_act']/df['cap_stk']
    growth_yoy_df(df[['cash_flow_per_share']],'cash_flow_per_share')


# ROE 增长率




# if __name__=='__main__':
#     fstrs=[f for f in locals().keys() if (f.startswith('get') and f!='get_ipython')]
#     for f in fstrs:#TODO:
#         eval(f)()
#         print(f)



#TODO: use a small sample to test the all the code
