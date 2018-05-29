# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-29  00:09
# NAME:FT-calculate_factors_new.py

import pandas as pd
from data.dataApi import read_local
from singleFactor.factors.base_function import x_pct_chg, x_compound_growth, \
    raw_level, ratio_yoy_pct_chg
from singleFactor.factors.check import check_factor

def check_raw_level(tbname,col,name):
    df=read_local(tbname)
    r_ttm=raw_level(df, col, ttm=True)
    r=raw_level(df, col, ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))

def check_g_yoy(tbname, col, name,q=4):
    '''
    yoy 增长率
    Args:
        tbname:
        col: 要检验的指标
        name: 保存的文件夹名
        q:int,q=4 表示yoy
    '''
    df=read_local(tbname)
    r_ttm=x_pct_chg(df,col,q=q,ttm=True)
    r=x_pct_chg(df,col,q=q,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,name)

def check_compound_g_yoy(tbname,col,name,q=60):
    '''
    复合增长率
    Args:
        tbname:
        col:
        name:
        q:

    Returns:

    '''
    df=read_local(tbname)
    r_ttm=x_compound_growth(df,col,q=q,ttm=True)
    r=x_compound_growth(df,col,q=q,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,name)


def check_ratio_yoy_pct_chg(tbnamex,colx,tbnamey,coly,name):
    '''
    yoy growth rate of x/y
    Args:
        tbnamex:
        colx:
        tbnamey:
        coly:
        name:

    Returns:

    '''
    x=read_local(tbnamex,colx)
    y=read_local(tbnamey,coly)
    df=pd.concat([x,y],axis=1)
    r_ttm=ratio_yoy_pct_chg(df,colx,coly,ttm=True)
    r=ratio_yoy_pct_chg(df,colx,coly,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,name)

#==============================================================================
def get_saleEarnings_sq_yoy():
    # 单季度营业利润同比增长率 saleEarnings_sq_yoy
    name='saleEarnings_sq_yoy'
    tbname = 'equity_selected_income_sheet_q'
    col = 'oper_profit'
    check_g_yoy(tbname, col, name)

def get_earnings_sq_yoy():
    #单季度净利润同比增长率 earnings_sq_yoy
    name='earnings_sq_yoy'
    tbname = 'equity_selected_income_sheet_q'
    col = 'net_profit_excl_min_int_inc'
    check_g_yoy(tbname, col, name)

def get_sales_sq_yoy():
    #单季度营业收入同比增长率 sales_sq_yoy
    name='sales_sq_yoy'
    tbname='equity_selected_income_sheet_q'
    col='oper_rev'
    check_g_yoy(tbname, col, name)

def get_eps1Ygrowth_yoy():
    #每股收益同比增长率 eps1Ygrowth_yoy
    name='esp1Ygrowth_yoy'
    tbnamex='equity_selected_income_sheet_q'
    colx='net_profit_excl_min_int_inc'
    tbnamey='equity_selected_balance_sheet_q'
    coly='cap_stk'
    check_ratio_yoy_pct_chg(tbnamex,colx,tbnamey,coly,name)

def get_ocfGrowth_yoy():
    #经营现金流增长率 ocfGrowth_yoy
    name='ocfGrowth_yoy'
    tbname='equity_selected_cashflow_sheet_q'
    col='net_cash_flows_oper_act'
    check_g_yoy(tbname,col,name)

def get_earnings_ltg():
    #净利润过去 5 年历史增长率 earnings_ltg
    name='earnings_ltg'
    tbname='equity_selected_income_sheet_q'
    col='net_profit_excl_min_int_inc'
    check_g_yoy(tbname,col,name,q=20)

def get_sales_ltg():
    # 营业收入过去 5 年历史增长率 sales_ltg
    name='sales_ltg'
    tbname = 'equity_selected_income_sheet_q'
    col = 'oper_rev'
    check_g_yoy(tbname,col,name,q=20)

def get_g_netCashFlow():
    #净现金流增长率 g_netCashFlow
    name='g_netCashFlow'
    tbname = 'equity_selected_cashflow_sheet_q'
    col = 'net_cash_flows_oper_act'
    check_g_yoy(tbname,col,name)

def get_g_netProfit12Qavg():
    #过去 12 个季度净利润平均年增长率 g_netProfit12Qavg
    '''等价于过去12个月的增长率'''
    name='g_netProfit12Qavg'
    tbname = 'equity_selected_income_sheet_q'
    col = 'net_profit_excl_min_int_inc'
    check_g_yoy(tbname,col,name,q=12)

def get_g_totalOperatingRevenue12Qavg():
    #过去 12 个季度营业总收入平均年增长率
    '''等价于过去12个月的增长率'''
    name='g_totalOperatingRevenue12Qavg'
    tbname='equity_selected_income_sheet_q'
    col='tot_oper_rev'
    check_g_yoy(tbname,col,name,q=12)

def get_g_totalAssets():
    #总资产增长率
    name='g_totalAssets'
    tbname='equity_selected_balance_sheet'
    col='tot_assets'
    check_g_yoy(tbname,col,name)

def get_g_epscagr5():
    #EPS 5年复合增长率
    name='g_epscagr5'
    tbname='equity_selected_income_sheet_q'
    col='oper_profit'
    check_compound_g_yoy(tbname,col,name,q=60)

def get_netOperateCashFlowPerShare():
    #每股经营活动净现金流增长率
    name='netOperateCashFlowPerShare'
    tbnamex='equity_selected_income_sheet_q'
    colx='net_cash_flows_oper_act'
    tbnamey='equity_selected_balance_sheet'
    coly='cap_stk'
    check_ratio_yoy_pct_chg(tbnamex,colx,tbnamey,coly,name)

def get_roe_growth_rate():
    # ROE 增长率
    name='g_roe'
    tbname = 'asharefinancialindicator'
    col='s_fa_roe'
    check_g_yoy(tbname,col,name)


#===================================质量因子====================================
def get_artRate():
    # 应收账款周转率
    name='artRate'
    tbname = 'asharefinancialindicator'
    col='s_fa_arturn'
    check_raw_level(tbname,col,name)

def get_cashRateOfSales():
    # 经营活动产生的现金流量净额/营业收入
    name='cashRateOfSales'
    tbnamex='equity_selected_income_sheet_q'
    colx='net_cash_flows_oper_act'
    tbnamey='equity_selected_cash_sheet_q'
    coly='oper_rev'
    check_ratio_yoy_pct_chg(tbnamex,colx,tbnamey,coly,name)

#TODO: 对于累计的指标，不能使用ttm ,检查那些来自累计报表的指标
def get_cashToCurrentLiability():
    #经营活动产生现金流量净额/流动负债
    name='cashToCurrentLiability'
    tbnamex = 'equity_selected_income_sheet_q'
    colx = 'net_cash_flows_oper_act'
    tbnamey = 'equity_selected_balance_sheet'
    coly = 'tot_cur_liab'
    check_ratio_yoy_pct_chg(tbnamex, colx, tbnamey, coly, name)

def get_cashToLiability():
    #经营活动产生现金流量净额/负债合计
    name='cashToLiability'
    tbnamex = 'equity_selected_income_sheet_q'
    colx = 'net_cash_flows_oper_act'
    tbnamey = 'equity_selected_balance_sheet'
    coly = 'tot_liab'
    check_ratio_yoy_pct_chg(tbnamex, colx, tbnamey, coly, name)

def get_currentAssetsToAsset():
    #流动资产/总资产
    name='currentAssetsToAsset'
    tbnamex='equity_selected_balance_sheet'
    colx='tot_cur_assets'
    tbnamey='equity_selected_balance_sheet'
    coly='tot_assets'
    check_ratio_yoy_pct_chg(tbnamex, colx, tbnamey, coly, name)

def get_currentRatio():
    #流动资产/总资产
    name='currentRatio'
    tbnamex='equity_selected_balance_sheet'
    colx='tot_cur_assets'
    tbnamey='equity_selected_balance_sheet'
    coly='tot_cur_liab'
    check_ratio_yoy_pct_chg(tbnamex, colx, tbnamey, coly, name)

def get_debtAssetsRatio():
    #总负债/总资产
    name='debtAssetsRatio'
    tbnamex='equity_selected_balance_sheet'
    colx='tot_liab'
    tbnamey='equity_selected_balance_sheet'
    coly='tot_assets'
    check_ratio_yoy_pct_chg(tbnamex, colx, tbnamey, coly, name)




