# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-29  00:09
# NAME:FT-calculate_factors_new.py

import pandas as pd
from data.dataApi import read_local
from singleFactor.factors.base_function import x_pct_chg, x_compound_growth, level
from singleFactor.factors.check import check_factor


def get_saleEarnings_sq_yoy():
    # 单季度营业利润同比增长率 saleEarnings_sq_yoy
    tbname = 'equity_selected_income_sheet_q'
    col = 'oper_profit'
    indicator=read_local(tbname)
    r_ttm=x_pct_chg(indicator,col,q=4,ttm=True)
    r=x_pct_chg(indicator,col,q=4,ttm=False)
    check_factor(r_ttm,'saleEarnings_sq_yoy_ttm')
    check_factor(r,'saleEarnings_sq_yoy')


def get_earnings_sq_yoy():
    #单季度净利润同比增长率 earnings_sq_yoy
    tbname = 'equity_selected_income_sheet_q'
    col = 'net_profit_excl_min_int_inc'
    indicator = read_local(tbname)
    r_ttm = x_pct_chg(indicator, col, q=4, ttm=True)
    r = x_pct_chg(indicator, col, q=4, ttm=False)
    check_factor(r_ttm, 'earnings_sq_yoy_ttm')
    check_factor(r, 'earnings_sq_yoy')

def get_sales_sq_yoy():
    #单季度营业收入同比增长率 sales_sq_yoy
    tbname='equity_selected_income_sheet_q'
    col='oper_rev'
    indicator = read_local(tbname)
    r_ttm = x_pct_chg(indicator, col, q=4, ttm=True)
    r = x_pct_chg(indicator, col, q=4, ttm=False)
    check_factor(r_ttm, 'sales_sq_yoy_ttm')
    check_factor(r, 'sales_sq_yoy')

def get_eps1Ygrowth_yoy():
    #每股收益同比增长率 eps1Ygrowth_yoy
    cap_stk=read_local('equity_selected_balance_sheet_q','cap_stk')
    income=read_local('equity_selected_income_sheet_q','net_profit_excl_min_int_inc')
    df=pd.concat([cap_stk,income],axis=1)
    df['income_per_share']=df['net_profit_excl_min_int_inc']/df['cap_stk']
    r_ttm=x_pct_chg(df,'income_per_share',q=4,ttm=True)
    r=x_pct_chg(df,'income_per_share',q=4,ttm=False)
    check_factor(r_ttm,'eps1Ygrowth_yoy_ttm')
    check_factor(r,'eps1Ygrowth_yoy')

def get_ocfGrowth_yoy():
    #经营现金流增长率 ocfGrowth_yoy
    tbname='equity_selected_cashflow_sheet_q'
    col='net_cash_flows_oper_act'
    indicator = read_local(tbname)
    r_ttm = x_pct_chg(indicator, col, q=4, ttm=True)
    r = x_pct_chg(indicator, col, q=4, ttm=False)
    check_factor(r_ttm, 'ocfGrowth_yoy_ttm')
    check_factor(r, 'ocfGrowth_yoy_yoy')

def get_earnings_ltg():
    #净利润过去 5 年历史增长率 earnings_ltg
    tbname='equity_selected_income_sheet_q'
    col='net_profit_excl_min_int_inc'
    indicator = read_local(tbname)
    r_ttm = x_pct_chg(indicator, col, q=20, ttm=True)
    r = x_pct_chg(indicator, col, q=20, ttm=False)
    check_factor(r_ttm, 'earnings_ltg_ttm')
    check_factor(r, 'earnings_ltg_yoy')

def get_sales_ltg():
    # 营业收入过去 5 年历史增长率 sales_ltg
    tbname = 'equity_selected_income_sheet_q'
    col = 'oper_rev'
    indicator=read_local(tbname)
    r_ttm = x_pct_chg(indicator, col, q=20, ttm=True)
    r = x_pct_chg(indicator, col, q=20, ttm=False)
    check_factor(r_ttm, 'sales_ltg_ttm')
    check_factor(r, 'sales_ltg_yoy')

def get_g_netCashFlow():
    #净现金流增长率 g_netCashFlow
    tbname = 'equity_selected_cashflow_sheet_q'
    col = 'net_cash_flows_oper_act'
    indicator = read_local(tbname)
    r_ttm = x_pct_chg(indicator, col, q=4, ttm=True)
    r = x_pct_chg(indicator, col, q=4, ttm=False)
    check_factor(r_ttm,'g_netCashFlow_ttm')
    check_factor(r,'g_netCashFlow')

def get_g_netProfit12Qavg():
    #过去 12 个季度净利润平均年增长率 g_netProfit12Qavg
    '''等价于过去12个月的增长率'''
    name='g_netProfit12Qavg'
    tbname = 'equity_selected_income_sheet_q'
    col = 'net_profit_excl_min_int_inc'
    indicator=read_local(tbname,col)
    r_ttm = x_pct_chg(indicator, col, q=12, ttm=True)
    r = x_pct_chg(indicator, col, q=12, ttm=False)
    check_factor(r_ttm, '{}_ttm'.format(name))
    check_factor(r, name)

def get_g_totalOperatingRevenue12Qavg():
    #过去 12 个季度营业总收入平均年增长率
    name='g_totalOperatingRevenue12Qavg'
    tbname='equity_selected_income_sheet_q'
    col='tot_oper_rev'
    indicator = read_local(tbname, col)
    r_ttm = x_pct_chg(indicator, col, q=12, ttm=True)
    r = x_pct_chg(indicator, col, q=12, ttm=False)
    check_factor(r_ttm, '{}_ttm'.format(name))
    check_factor(r, name)

def get_g_totalAssets():
    #总资产增长率
    name='g_totalAssets'
    tbname='equity_selected_balance_sheet'
    col='tot_assets'
    indicator = read_local(tbname)
    r_ttm = x_pct_chg(indicator, col, q=4, ttm=True)
    r = x_pct_chg(indicator, col, q=4, ttm=False)
    check_factor(r_ttm, '{}_ttm'.format(name))
    check_factor(r, name)

def get_g_epscagr5():
    #EPS 5年复合增长率
    name='g_epscagr5'
    tbname='equity_selected_income_sheet_q'
    col='oper_profit'
    indicator = read_local(tbname)
    r_ttm=x_compound_growth(indicator,col,q=60,ttm=True)
    r=x_compound_growth(indicator,col,q=60,ttm=False)
    check_factor(r_ttm, '{}_ttm'.format(name))
    check_factor(r, name)

def get_netOperateCashFlowPerShare():
    #每股经营活动净现金流增长率
    name='netOperateCashFlowPerShare'
    cash_flow=read_local('equity_selected_income_sheet_q','net_cash_flows_oper_act')
    cap_stk=read_local('equity_selected_balance_sheet_q','cap_stk')
    df=pd.concat([cap_stk,cash_flow],axis=1)
    df['result']=df['net_cash_flows_oper_act']/df['cap_stk']
    r_ttm=x_pct_chg(df,'result',q=4,ttm=True)
    r=x_pct_chg(df,'result',q=4,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))

def get_roe_growth_rate():
    # ROE 增长率
    name='g_roe'
    tbname = 'asharefinancialindicator'
    col='s_fa_roe'
    df=read_local(tbname)
    r_ttm=x_pct_chg(df,col,q=4,ttm=True,delete_negative=False)
    r=x_pct_chg(df,col,q=4,ttm=False,delete_negative=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))




#===================================质量因子====================================
def get_artRate():
    # 应收账款周转率
    name='artRate'
    tbname = 'asharefinancialindicator'
    col='s_fa_arturn'
    df=read_local(tbname)
    r_ttm=level(df,col,ttm=True)
    r=level(df,col,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))

def get_cashRateOfSales():
    # 经营活动产生的现金流量净额/营业收入
    name='cashRateOfSales'
    cash_flow=read_local('equity_selected_income_sheet_q','net_cash_flows_oper_act')
    oper_rev=read_local('equity_selected_cash_sheet_q','oper_rev')
    df=pd.concat([oper_rev,cash_flow],axis=1)
    df['result']=df['net_cash_flows_oper_act']/df['oper_rev']
    r_ttm=x_pct_chg(df,'result',q=4,ttm=True)
    r=x_pct_chg(df,'result',q=4,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))

#TODO: 对于累计的指标，不能使用ttm ,检查那些来自累计报表的指标
def get_cashToCurrentLiability():
    #经营活动产生现金流量净额/流动负债
    name='cashToCurrentLiability'
    cash_flow=read_local('equity_selected_income_sheet','net_cash_flows_oper_act')
    cur_liab=read_local('equity_selected_balance_sheet','tot_cur_liab')
    df=pd.concat([cur_liab,cash_flow],axis=1)
    df['result']=df['net_cash_flows_oper_act']/df['tot_cur_liab']
    r_ttm=x_pct_chg(df,'result',q=4,ttm=True)
    r=x_pct_chg(df,'result',q=4,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))

def get_cashToLiability():
    #经营活动产生现金流量净额/负债合计
    name='cashToLiability'
    cash_flow=read_local('equity_selected_income_sheet','net_cash_flows_oper_act')
    cur_liab=read_local('equity_selected_balance_sheet','tot_liab')
    df=pd.concat([cur_liab,cash_flow],axis=1)
    df['result']=df['net_cash_flows_oper_act']/df['tot_liab']
    r_ttm=x_pct_chg(df,'result',q=4,ttm=True)
    r=x_pct_chg(df,'result',q=4,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))

def get_currentAssetsToAsset():
    #流动资产/总资产
    name='currentAssetsToAsset'
    cur_assets=read_local('equity_selected_balance_sheet','tot_cur_assets')
    tot_assets=read_local('equity_selected_balance_sheet','tot_assets')
    df = pd.concat([cur_assets, tot_assets], axis=1)
    df['result'] = df['tot_cur_assets'] / df['tot_assets']
    r_ttm = x_pct_chg(df, 'result', q=4, ttm=True)
    r = x_pct_chg(df, 'result', q=4, ttm=False)
    check_factor(r_ttm, '{}_ttm'.format(name))
    check_factor(r, '{}'.format(name))

