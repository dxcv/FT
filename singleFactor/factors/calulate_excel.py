# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-04  09:19
# NAME:FT-calulate_excel.py

import multiprocessing
import os
import pickle

import pandas as pd
import numpy as np

from data.dataApi import read_local_pkl, get_dataspace

from singleFactor.factors.base_function import x_history_growth_avg, \
    x_history_compound_growth, ttm_adjust, x_history_downside_std
from singleFactor.factors.cal_tools import get_dataspace_old, check_g_yoy, \
    check_ratio_yoy_pct_chg, check_raw_level, check_compound_g_yoy, check_ratio, \
    check_stability, check_level_square
from singleFactor.factors.check import check_factor, _check

#===============================成长因子========================================
def get_saleEarnings_sq_yoy():
    # 单季度营业利润同比增长率 saleEarnings_sq_yoy
    name='G_saleEarnings_sq_yoy'
    df=get_dataspace_old('oper_profit')
    check_g_yoy(df,'oper_profit',name)

def get_earnings_sq_yoy():
    #单季度净利润同比增长率 earnings_sq_yoy
    name='G_earnings_sq_yoy'
    col = 'net_profit_excl_min_int_inc'
    df=get_dataspace_old(col)
    check_g_yoy(df, col, name)

def get_sales_sq_yoy():
    #单季度营业收入同比增长率 sales_sq_yoy
    name='G_sales_sq_yoy'
    col='oper_rev'
    df=get_dataspace_old(col)
    check_g_yoy(df, col, name)

def get_eps1Ygrowth_yoy():
    #每股收益同比增长率 eps1Ygrowth_yoy
    name='G_esp1Ygrowth_yoy'
    df=get_dataspace_old(['net_profit_excl_min_int_inc', 'cap_stk'])
    check_ratio_yoy_pct_chg(df,'net_profit_excl_min_int_inc','cap_stk',name)

def get_ocfGrowth_yoy():
    #经营现金流增长率 ocfGrowth_yoy
    name='G_ocfGrowth_yoy'
    col='net_cash_flows_oper_act'
    df=get_dataspace_old(col)
    check_g_yoy(df,col,name)

def get_earnings_ltg():
    #净利润过去 5 年历史增长率 earnings_ltg
    name='G_earnings_ltg'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace_old(col)
    check_g_yoy(df,col,name,q=20)

def get_sales_ltg():
    # 营业收入过去 5 年历史增长率 sales_ltg
    name='G_sales_ltg'
    col = 'oper_rev'
    df=get_dataspace_old(col)
    check_g_yoy(df,col,name,q=20)

def get_g_netCashFlow():
    #净现金流增长率 g_netCashFlow
    name='G_netCashFlow'
    col = 'net_cash_flows_oper_act'
    df=get_dataspace_old(col)
    check_g_yoy(df,col,name)

def get_g_netProfit12Qavg():
    #过去 12 个季度净利润平均年增长率 g_netProfit12Qavg
    name='G_netProfit12Qavg'
    col = 'net_profit_excl_min_int_inc'
    df=get_dataspace_old(col)
    df=x_history_growth_avg(df,col,q=12)
    check_raw_level(df,'target',name)

#TODO:define a standard function for this type of indicators
def get_g_totalOperatingRevenue12Qavg():
    #过去 12 个季度营业总收入平均年增长率
    name='G_totalOperatingRevenue12Qavg'
    col='tot_oper_rev'
    df=get_dataspace_old(col)
    df=x_history_growth_avg(df,col,q=12)
    check_raw_level(df,'target',name)

def get_g_totalAssets():
    #总资产增长率
    name='G_totalAssets'
    col='tot_assets'
    df=get_dataspace_old(col)
    check_g_yoy(df,col,name)

def get_g_epscagr5():
    #EPS 5年复合增长率
    name='G_epscagr5'
    col='oper_profit'
    df=get_dataspace_old(col)
    check_compound_g_yoy(df,col,name,q=20)

def get_netOperateCashFlowPerShare():
    #每股经营活动净现金流增长率
    name='G_netOperateCashFlowPerShare'
    colx='net_cash_flows_oper_act'
    coly='cap_stk'
    df=get_dataspace_old([colx, coly])
    check_ratio_yoy_pct_chg(df,colx,coly,name)

def get_roe_growth_rate():#TODO: add indicators
    # ROE 增长率
    name='G_roe'
    col='s_fa_roe'
    df=get_dataspace_old(col)
    check_g_yoy(df,col,name)


#TODO: add equity_cash_dividend to get_dataspace_old
def get_dividend3YR():
    #股息3年复合增长率
    name='G_dividend3YR'
    df1=read_local_pkl('equity_cash_dividend')
    df2=get_dataspace_old('tot_assets')
    df1['target']=df1['cash_div'].groupby('stkcd').pct_change(3).replace([-np.inf,np.inf],0)
    df=pd.concat([df1[['target','trd_dt']],df2[['tot_assets','trd_dt']]],axis=1)
    check_factor(df,name)

def get_g_NetProfit():
    #净利润增长率
    name='G_NetProfit'
    df=get_dataspace_old('net_profit_excl_min_int_inc')
    check_g_yoy(df,'net_profit_excl_min_int_inc',name)

def get_NetProfitCAGR3():
    #净利润 3 年复合增长率
    name='G_netProfitCAGR3'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace_old(col)
    df=x_history_compound_growth(df, col, q=12)
    check_raw_level(df,'target',name)

def get_NetProfitCAGR5():
    #净利润 5 年复合增长率
    name='G_netProfitCAGR5'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace_old(col)
    df=x_history_compound_growth(df, col,q=20)
    check_raw_level(df,'target',name)

def get_g_netAssetsPerShare():
    #每股净资产增长率
    name='G_netAssetsPerShare'
    df=get_dataspace_old(['tot_assets', 'tot_liab'])
    df['net_asset']=df['tot_assets']-df['tot_liab']
    check_g_yoy(df,'net_asset',name)

def get_g_operatingRevenueCAGR3():
    #营业收入 3 年复合增长率
    name='G_operatingRevenueCAGR3'
    df=get_dataspace_old('oper_rev')
    df=x_history_compound_growth(df, 'oper_rev', q=12)
    check_raw_level(df,'target',name)

def get_g_operatingRevenueCAGR5():
    #营业收入 5 年复合增长率
    name='G_operatingRevenueCAGR5'
    df=get_dataspace_old('oper_rev')
    df=x_history_compound_growth(df, 'oper_rev', q=20)
    check_raw_level(df,'target',name)

def get_g_totalOperatingRevenue():
    #营业总收入增长率
    name='G_totalOperatingRevenue'
    col='tot_oper_rev'
    df=get_dataspace_old(col)
    check_g_yoy(df,col,name)

def get_netProfit3YRAvg():
    #3 年净利润增长率的平均值
    name='G_netProfit3YRAvg'
    df=get_dataspace_old('net_profit_excl_min_int_inc')
    df=x_history_growth_avg(df,'net_profit_excl_min_int_inc',q=12)
    check_raw_level(df,'target',name)

#===================================质量因子====================================
def get_artRate(): #TODO: use asharefinancialindicator
    return
    # 应收账款周转率
    name='Q_artRate'
    tbname = 'asharefinancialindicator'
    col='s_fa_arturn'
    check_raw_level(tbname,col,name)


#TODO: 注意 除法 容易出现inf 和 -inf
def get_cashDividendCover():
    #现金股利保障倍数＝经营活动产生的现金流量净额/累计合计派现金额
    name='Q_cashDividendCover'
    df=get_dataspace_old(['net_cash_flows_oper_act', 'cash_div'])
    df['target']=df['net_cash_flows_oper_act']/df['cash_div']
    check_raw_level(df,'target',name)

# get_cashDividendCover()

def get_cashRateOfSales():
    # 经营活动产生的现金流量净额/营业收入
    name='Q_cashRateOfSales'
    cols=['net_cash_flows_oper_act','oper_rev']
    df=get_dataspace_old(cols)
    check_ratio(df,cols[0],cols[1],name)

#TODO: 对于累计的指标，不能使用ttm ,检查那些来自累计报表的指标
def get_cashToCurrentLiability():
    #经营活动产生现金流量净额/流动负债
    name='Q_cashToCurrentLiability'
    cols=['net_cash_flows_oper_act','tot_cur_liab']
    df=get_dataspace_old(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_cashToLiability():
    #经营活动产生现金流量净额/负债合计
    name='Q_cashToLiability'
    cols=['net_cash_flows_oper_act','tot_liab']
    df=get_dataspace_old(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_currentAssetsToAsset():
    #流动资产/总资产
    name='Q_currentAssetsToAsset'
    cols=['tot_cur_assets','tot_assets']
    df=get_dataspace_old(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_currentRatio():
    #流动资产/流动负债
    name='Q_currentRatio'
    cols=['tot_cur_assets','tot_cur_liab']
    df=get_dataspace_old(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_debtAssetsRatio():
    #总负债/总资产
    name='Q_debtAssetsRatio'
    cols=['tot_liab','tot_assets']
    df=get_dataspace_old(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_dividendCover(): #TODO: dividend
    #股息保障倍数＝归属于母公司的净利润/最近 1 年的累计派现金额
    name='Q_dividendCover'
    df=get_dataspace_old(['net_profit_excl_min_int_inc', 'cash_div'])
    df['target'] = df['net_profit_excl_min_int_inc'] / df['cash_div']
    check_raw_level(df, 'target', name,ttm=False)

def get_earningsStability():
    #净利润过去 2 年的标准差
    name='Q_earningsStability'
    col='net_profit_incl_min_int_inc'
    df=get_dataspace_old(col)
    check_stability(df,col,name)

def get_equityToAsset():
    #股东权益比率=股东权益合计/总资产
    name='Q_equityToAsset'
    colx='tot_shrhldr_eqy_excl_min_int'
    coly='tot_assets'
    df=get_dataspace_old([colx, coly])
    check_ratio(df,colx,coly,name)

def get_equityTurnover():
    #股东权益周转率＝营业总收入*2/(期初净资产+期末净资产)
    name='Q_equityTurnover'
    df=get_dataspace_old(['tot_oper_rev', 'tot_assets', 'tot_liab'])
    df['x']=df['tot_oper_rev']*2
    df['net_assets']=df['tot_assets']-df['tot_liab']
    df['y']=df['net_assets'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['target']=df['x']/df['y']
    check_raw_level(df,'target',name)

def get_fixedAssetTurnover():
    #营业收入 * 2  /  （期初固定资产 + 期末固定资产）
    name='Q_fixedAssetTrunover'
    df=get_dataspace_old(['tot_oper_rev', 'fix_assets'])
    df['x']=df['tot_oper_rev']*2
    df['y']=df['fix_assets'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['target']=df['x']/df['y']
    check_raw_level(df,'target',name)

def get_grossIncomeRatio():
    #销售毛利率=[营业收入-营业成本]/营业收入
    name='Q_grossIncomeRatio'
    df=get_dataspace_old(['oper_rev', 'oper_cost'])
    df['target']=(df['oper_rev']-df['oper_cost'])/df['oper_rev']
    check_raw_level(df,'target',name)

def get_intanibleAssetRatio():
    #无形资产比率=无形资产/总资产
    name='Q_intangibleAssetRatio'
    cols=['intang_assets','tot_assets']
    df=get_dataspace_old(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_interestCover():
    return
    #TODO:int_exp 数据缺失严重
    #利息保障倍数＝息税前利润/利息费用
    '''
    息税前利润=净利润+所得税+财务费用
    '''
    name='Q_interestCover'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    df=get_dataspace_old([col1, col2, col3])
    df['x']=df[col1]+df[col2]+df[col3]
    df['target']=df['x']/df['int_exp']
    check_raw_level(df,'target',name)

def get_inventoryTurnover():
    #营业成本 * 2  /  （期初存货净额 + 期末存货净额）
    name='Q_inventoryTrunover'
    df=get_dataspace_old(['oper_rev', 'inventories'])
    df['x']=df['oper_rev']*2
    df['y']=df['inventories'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['target']=df['x']/df['y']
    check_raw_level(df,'target',name)

def get_mlev(): #TODO:
    #长期负债/(长期负债+市值)
    '''
    负债=流动负债+非流动负债=短期负债+长期负债
    流动负债=短期负债
    长期负债=非流动负债=长期借款+应付债券+长期应付款
    '''
    name='Q_mlev'
    df1=read_local_pkl('equity_fundamental_info')
    df2=read_local_pkl('equity_selected_balance_sheet')
    df2=df2.reset_index().sort_values(['stkcd','trd_dt','report_period'])
    df2=df2[~df2.duplicated(subset=['stkcd','trd_dt'],keep='last')].set_index(['stkcd','trd_dt'])
    df=pd.concat([df2,df1],join='inner',axis=1).reset_index()
    df=df.set_index(['stkcd','report_period']).sort_index()
    df['target']=df['tot_non_cur_liab']/(df['tot_non_cur_liab']+df['freefloat_cap'])
    check_raw_level(df,'target',name)

def get_netNonOItoTP():
    #营业外收支净额/利润总额
    name='Q_netNonOItoTP'
    df=get_dataspace_old(['non_oper_rev', 'non_oper_exp', 'net_profit_excl_min_int_inc'])
    df['target']=(df['non_oper_rev']-df['non_oper_exp'])/df['net_profit_excl_min_int_inc']
    check_raw_level(df,'target',name)

def get_netProfitCashCover():
    #经营活动产生的现金流量净额/净利润
    name='Q_netProfitCashCover'
    df=get_dataspace_old(['net_cash_flows_oper_act', 'net_profit_excl_min_int_inc'])
    df['target']=df['net_cash_flows_oper_act']/df['net_profit_excl_min_int_inc']
    check_raw_level(df,'target',name)

def get_NPCutToNetRevenue():#TODO:数据缺失
    #扣除非经常损益后的净利润/营业总收入
    name='Q_NPCutToNetRevenue'
    df=get_dataspace_old(['net_profit_after_ded_nr_lp', 'tot_oper_rev'])
    df['target']=df['net_profit_after_ded_nr_lp']/df['tot_oper_rev']
    check_raw_level(df,'target',name)

def get_netProfitRatio():
    #销售净利率＝含少数股东损益的净利润/营业收入
    name='Q_netProfitRatio'
    col1='net_profit_incl_min_int_inc'
    col2='tot_oper_rev'
    df=get_dataspace_old([col1, col2])
    df['target']=df[col1]/df[col2]
    check_raw_level(df,'target',name)

def get_netProfitTorevenue():
    #净利润/营业总收入
    name='Q_netProfitToRevenue'
    colx='net_profit_excl_min_int_inc'
    coly='tot_oper_rev'
    df=get_dataspace_old([colx, coly])
    check_ratio(df,colx,coly,name)

def get_netProfitToTotProfit():
    #净利润/利润总额
    name='Q_netProfitToTotProfit'
    colx='net_profit_excl_min_int_inc'
    coly='oper_profit'
    df=get_dataspace_old([colx, coly])
    check_ratio(df,colx,coly,name)

def get_NPCutToNetProfit():
    #扣除非经常损益后的净利润/归属于母公司的净利润
    return
    name='Q_NPCutToNetProfit'
    colx='net_profit_after_ded_nr_lp' #TODO: 数据缺失严重
    coly='net_profit_excl_min_int_inc'
    df=get_dataspace_old([colx, coly])
    check_ratio(df,colx,coly,name)

def get_operatingExpenseRate():
    #销售费用/营业总收入
    name='Q_operatingExpenseRate'
    colx = 'selling_dist_exp'
    coly = 'tot_oper_rev'
    df=get_dataspace_old([colx, coly])
    check_ratio(df,colx,coly,name)

def get_operatingProfitToAsset():
    #营业利润/总资产
    name='Q_operatingProfitToAsset'
    colx='oper_profit'
    coly='tot_assets'
    df=get_dataspace_old([colx, coly])
    check_ratio(df,colx,coly,name)

def get_operatingProfitToEquity():
    #TODO: 所有者权益
    #营业利润/净资产
    name='Q_operatingProfitToEquity'
    df=get_dataspace_old(['oper_profit', 'tot_assets', 'tot_liab'])
    df['target']=df['oper_profit']/(df['tot_assets']-df['tot_liab'])
    check_raw_level(df,'target',name)

def get_operCashInToAsset():
    #总资产现金回收率＝经营活动产生的现金流量净额 * 2/(期初总资产+期末总资产)
    name='Q_operCashInToAsset'
    df=get_dataspace_old(['net_cash_flows_oper_act', 'tot_assets'])
    df['x']=df['net_cash_flows_oper_act']*2
    df['y']=df['tot_assets'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['target']=df['x']/df['y']
    check_raw_level(df,'target',name)

def get_operCashInToCurrentDebt():
    #现金流动负债比=经营活动产生的现金流量净额/流动负债
    name='Q_operCashInToCurrentDebt'
    df=get_dataspace_old(['net_cash_flows_oper_act', 'oper_rev'])
    df['target']=df['net_cash_flows_oper_act']/df['oper_rev']
    check_raw_level(df,'target',name)

def get_periodCostsRate():
    #销售期间费用率＝[营业费用+管理费用+财务费用]/营业收入
    name='Q_periodCostsRate'
    df=get_dataspace_old(['oper_cost', 'gerl_admin_exp', 'fin_exp', 'tot_oper_cost'])
    df['target']=(df['oper_cost']+df['gerl_admin_exp']+df['fin_exp'])/df['tot_oper_cost']
    check_raw_level(df,'target',name)

# get_periodCostsRate()

def get_quickRatio():
    #速动比率＝(流动资产合计-存货)/流动负债合计
    name='Q_quickRatio'
    df=get_dataspace_old(['tot_cur_assets', 'inventories', 'tot_cur_liab'])
    df['target']=(df['tot_cur_assets']-df['inventories'])/df['tot_cur_liab']
    check_raw_level(df,'target',name)

def get_receivableTopayable():
    #应收应付比 = （应收票据+应收账款） / （应付票据+应付账款）
    name='Q_receivableTopayble'
    df=get_dataspace_old(['notes_rcv', 'acct_rcv', 'notes_payable', 'acct_payable'])
    df['target']=(df['notes_rcv']+df['acct_rcv'])/(df['notes_payable']+df['acct_payable'])
    check_raw_level(df,'target',name)

def get_roa():
    #总资产净利率=净利润(含少数股东损益)TTM/总资产
    name='Q_roa'
    df=get_dataspace_old(['net_profit_incl_min_int_inc', 'tot_assets'])
    df['ttm']=ttm_adjust(df['net_profit_incl_min_int_inc'])
    df['target']=df['ttm']/df['tot_assets']
    check_raw_level(df,'target',name)

def get_roe_ebit():
    #总资产报酬率＝息税前利润/总资产
    name='Q_roe_ebit'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    col4='tot_assets'
    df=get_dataspace_old([col1, col2, col3, col4])
    df['target']=(df[col1]+df[col2]+df[col3])/df[col4]
    check_raw_level(df,'target',name)

def get_roa_ebit():
    #归属于母公司的净利润/归属于母公司的股东权益
    name='Q_roe'
    colx='net_profit_excl_min_int_inc'
    coly='tot_shrhldr_eqy_excl_min_int'
    df=get_dataspace_old([colx, coly])
    df['target']=df[colx]/df[coly]
    check_raw_level(df,'target',name)

def get_operatingCostToTOR():
    #营业总成本/营业总收入
    name='Q_operatingCostToTOR'
    colx = 'tot_oper_rev'
    coly = 'tot_oper_cost'
    df=get_dataspace_old([colx, coly])
    check_ratio(df,colx,coly,name)

def get_downturnRisk():
    #std(min(本季度现金流-上一季度现金流,0))
    name='Q_downturnRisk'
    df=get_dataspace_old('net_cash_flows_oper_act')
    df=x_history_downside_std(df,'net_cash_flows_oper_act',q=8)
    check_raw_level(df,'target',name)

#======================================价值因子=========================
def get_bp():
    #股东权益合计  /  总市值
    name='V_bp'
    col1='tot_shrhldr_eqy_excl_min_int'
    col2='tot_assets'
    df=get_dataspace_old([col1, col2])
    df['target']=df[col1]/df[col2]
    check_raw_level(df,'target',name)

def get_ep():
    #净利润(不含少数股东损益)  /  总市值
    name='V_ep'
    col1='net_profit_excl_min_int_inc'
    col2='tot_assets'
    df=get_dataspace_old([col1, col2])
    df['target']=df[col1]/df[col2]
    check_raw_level(df,'target',name)

def get_sp():
    #营业收入  /  总市值
    name='V_sp'
    col1='oper_rev'
    col2='tot_assets'
    df=get_dataspace_old([col1, col2])
    df['target']=df[col1]/df[col2]
    check_raw_level(df,'target',name)

def get_cfp():
    #经营活动产生的现金流量净额  /  总市值
    name='V_cfp'
    col1='net_cash_flows_oper_act'
    col2='tot_assets'
    df=get_dataspace_old([col1, col2])
    df['target']=df[col1]/df[col2]
    check_raw_level(df,'target',name)

def get_sales_EV():
    #营业收入/(总市值+非流动负债合计-货币资金)
    name='V_sales_EV'
    col1='oper_rev'
    col2='tot_assets'
    col3='tot_non_cur_liab'
    col4='monetary_cap'
    df=get_dataspace_old([col1, col2, col3, col4])
    df['target']=df[col1]/(df[col2]+df[col3]-df[col4])
    check_raw_level(df,'target',name)


def get_ebitda_p():
    #息税折旧摊销前利润  /  总市值
    name='V_ebitda_p'
    col1='ebitda'
    col2='cap'
    df=get_dataspace([col1, col2])
    df['target']=df[col1]/df[col2]
    check_raw_level(df,'target',name)

get_ebitda_p()



def get_peg_nY():
    #(1/EP)/n年历史复合增长率
    name='V_peg_nY'
    col1 = 'net_profit_excl_min_int_inc'
    col2 = 'tot_assets'
    df = get_dataspace_old([col1, col2])
    df['ep'] = df[col1] / df[col2]
    df['peg']=1/df['ep']
    check_compound_g_yoy(df,'peg',name)

def get_dp_new():
    name='V_dp'
    df1=read_local_pkl('equity_cash_dividend')
    df2=get_dataspace_old(['tot_assets'])

    df=pd.concat([df1[['cash_div','trd_dt']],df2[['tot_assets','trd_dt']]],axis=1)
    df['target']=df['cash_div']/df['tot_assets']
    check_factor(df,name)

def get_p_square():#TODO: add cap to dataspace,compare with get_mlev
    #市值平方
    name='V_p_square'
    df=read_local_pkl('equity_fundamental_info')
    df['target']=df['freeshares']*df['freeshares']
    _check(df[['target']],name)

#========================================unfiled================================
def get_ebitToTLiablity():
    #总资产报酬率＝息税前利润/总资产
    name='G_ebitToTLiability'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    col4='tot_oper_rev'
    df=get_dataspace_old([col1, col2, col3])
    df['target']=(df[col1]+df[col2]+df[col3])/df[col4]
    check_raw_level(df,'target',name)


#=============================================main==============================
#TODO: repalace the long table name with compact name
#TODO: operator (col1+col2-col3)*2/(col1+col1(-1))
#TODO: std(8,col1)

#TODO: send a freq parameter to determine which frequency to use.

def task(f):
    try:
        eval(f)()
    except Exception as e:
        with open(r'D:\zht\database\quantDb\internship\FT\singleFactor\failed1.txt','a') as txt:
            txt.write('{} ->  {}\n'.format(f,e))

#TODO:organiza the indicators in calculate+factors.py

# if __name__=='__main__':
#     # fstrs=[f for f in locals().keys() if (f.startswith('get') and f!='get_ipython')]
#     fstrs=[l.split(' ')[0] for l in open(r'D:\zht\database\quantDb\internship\FT\singleFactor\invalid_ids.txt').read().split('\n')]
#     pool=multiprocessing.Pool(2)
#     pool.map(task,fstrs)




'''
1. 统计每个时间截面的样本量
'''