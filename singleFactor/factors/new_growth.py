# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  08:54
# NAME:FT-new_growth.py
from config import SINGLE_D_INDICATOR_FINANCIAL
from data.dataApi import get_dataspace
import os
from singleFactor.factors.new_operators import x_pct_chg, ratio_pct_chg, \
    x_history_growth_avg, x_history_compound_growth


def save_indicator(df,name):
    df[['trd_dt',name]].to_pickle(os.path.join(SINGLE_D_INDICATOR_FINANCIAL, name + '.pkl'))

#===============================成长因子========================================
#TODO: 同比，表示yoy？
#TODO: 5年增长率

'''
增长率：
    history=1,4,12,20
    method=avg,compound,std,downside_std
    ttm=True,False
    
'''


#------------------------增长率------------------------
def get_G_saleEarnings_sq_yoy():
    # 单季度营业利润同比增长率 saleEarnings_sq_yoy
    name='G_saleEarnings_sq_yoy'
    col='oper_profit'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=4)
    save_indicator(df,name)

def get_G_earnings_sq_yoy():
    #单季度净利润同比增长率 earnings_sq_yoy
    name='G_earnings_sq_yoy'
    col = 'net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df[name] = x_pct_chg(df[col],q=4)
    save_indicator(df, name)

def get_G_sales_sq_yoy():
    #单季度营业收入同比增长率 sales_sq_yoy
    name='G_sales_sq_yoy'
    col='oper_rev'
    df = get_dataspace(col)
    df[name] = x_pct_chg(df[col],q=4)
    save_indicator(df, name)

def get_G_ocfGrowth_yoy():
    #经营现金流增长率 ocfGrowth_yoy
    name='G_ocfGrowth_yoy'
    col='net_cash_flows_oper_act'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=4)
    save_indicator(df,name)

def get_g_netCashFlow():
    #净现金流增长率 g_netCashFlow
    name='G_netCashFlow'
    col = 'net_cash_flows_oper_act'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=4)
    save_indicator(df,name)

def get_g_totalAssets():
    #总资产增长率
    name='G_totalAssets'
    col='tot_assets'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=4)
    save_indicator(df,name)

def get_roe_growth_rate():#TODO: add indicators
    # ROE 增长率
    name='G_roe'
    col='s_fa_roe'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=4)
    save_indicator(df,name)

def get_g_totalOperatingRevenue():
    #营业总收入增长率
    name='G_totalOperatingRevenue'
    col='tot_oper_rev'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=4)
    save_indicator(df,name)

def get_dividend3YR():
    #股息3年复合增长率
    name='G_dividend3YR'
    col='cash_div'
    df=get_dataspace(col)
    df[name]=x_history_compound_growth(df[col],q=12)
    save_indicator(df,name)

#-------------------3年增长率----------------------
def get_g_totalOperatingRevenue12Qavg():
    #过去 12 个季度营业总收入平均年增长率
    name='G_totalOperatingRevenue12Qavg'
    col='tot_oper_rev'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=12)
    save_indicator(df,name)

def get_netProfit3YRAvg():
    #3 年净利润增长率的平均值
    name='G_netProfit3YRAvg'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=12)
    save_indicator(df,name)

#------------------3 年复合增长率-----------------
def get_NetProfitCAGR3():
    #净利润 3 年复合增长率
    name='G_netProfitCAGR3'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df[name]=x_history_compound_growth(df[col],q=12)
    save_indicator(df,name)

def get_g_operatingRevenueCAGR3():
    #营业收入 3 年复合增长率
    name='G_operatingRevenueCAGR3'
    col='oper_rev'
    df=get_dataspace(col)
    df[name]=x_history_compound_growth(df[col],q=12)
    save_indicator(df,name)

#------------------5 年增长率-----------------
def get_earnings_ltg():#TODO:
    #净利润过去 5 年历史增长率 earnings_ltg
    name='G_earnings_ltg'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=20)
    save_indicator(df,name)

def get_sales_ltg():
    # 营业收入过去 5 年历史增长率 sales_ltg
    name='G_sales_ltg'
    col = 'oper_rev'
    df=get_dataspace(col)
    df[name]=x_pct_chg(df[col],q=20)
    save_indicator(df,name)

#---------------5年复合增长率-------------------
def get_g_epscagr5():
    #EPS 5年复合增长率
    name='G_epscagr5'
    col='oper_profit'
    df=get_dataspace(col)
    df[name]=x_history_compound_growth(df[col],q=20)
    save_indicator(df,name)

def get_NetProfitCAGR5():
    #净利润 5 年复合增长率
    name='G_netProfitCAGR5'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df[name]=x_history_compound_growth(df[col],q=20)
    save_indicator(df,name)

def get_g_operatingRevenueCAGR5():
    #营业收入 5 年复合增长率
    name='G_operatingRevenueCAGR5'
    col='oper_rev'
    df=get_dataspace(col)
    df[name]=x_history_compound_growth(df[col],q=20)
    save_indicator(df,name)



#------------- Two indicator---------------------
def get_G_esp1Ygrowth_yoy():
    #每股收益同比增长率 eps1Ygrowth_yoy
    name='G_esp1Ygrowth_yoy'
    x='net_profit_excl_min_int_inc'
    y='cap_stk'
    df=get_dataspace([x,y])
    df[name]=ratio_pct_chg(df, x, y, q=4)
    save_indicator(df,name)

def get_netOperateCashFlowPerShare():
    #每股经营活动净现金流增长率
    name='G_netOperateCashFlowPerShare'
    x='net_cash_flows_oper_act'
    y='cap_stk'
    df=get_dataspace([x, y])
    df[name]=ratio_pct_chg(df, x, y)
    save_indicator(df,name)

def get_g_netAssetsPerShare():
    #每股净资产增长率
    name='G_netAssetsPerShare'
    col1='tot_assets'
    col2='tot_liab'
    df=get_dataspace([col1,col2])
    df[name]=x_pct_chg(df[col1]-df[col2],q=4)
    x_pct_chg(df[name],q=4)


