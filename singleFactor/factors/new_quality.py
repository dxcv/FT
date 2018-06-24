# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  11:12
# NAME:FT-new_quality.py

import os

from config import SINGLE_D_INDICATOR_FINANCIAL
from data.dataApi import get_dataspace
from singleFactor.factors.new_operators import ratio, x_history_std, x_ttm, \
    x_history_downside_std


def save_indicator(df,name):
    df[['trd_dt',name]].to_pickle(os.path.join(SINGLE_D_INDICATOR_FINANCIAL, name + '.pkl'))


#------------------------------ratio------------------------------------
def get_cashDividendCover():
    #现金股利保障倍数＝经营活动产生的现金流量净额/累计合计派现金额
    name='Q_cashDividendCover'
    col1='net_cash_flows_oper_act'
    col2='cash_div'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_cashRateOfSales():
    # 经营活动产生的现金流量净额/营业收入
    name='Q_cashRateOfSales'
    col1='net_cash_flows_oper_act'
    col2='oper_rev'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_cashToCurrentLiability():
    #经营活动产生现金流量净额/流动负债
    name='Q_cashToCurrentLiability'
    col1='net_cash_flows_oper_act'
    col2='tot_cur_liab'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_cashToLiability():
    #经营活动产生现金流量净额/负债合计
    name='Q_cashToLiability'
    col1='net_cash_flows_oper_act'
    col2='tot_liab'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_currentAssetsToAsset():
    #流动资产/总资产
    name='Q_currentAssetsToAsset'
    col1='tot_cur_assets'
    col2='tot_assets'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_currentRatio():
    #流动资产/流动负债
    name='Q_currentRatio'
    col1='tot_cur_assets'
    col2='tot_cur_liab'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_debtAssetsRatio():
    #总负债/总资产
    name='Q_debtAssetsRatio'
    col1='tot_liab'
    col2='tot_assets'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_intanibleAssetRatio():
    #无形资产比率=无形资产/总资产
    name='Q_intangibleAssetRatio'
    col1='intang_assets'
    col2='tot_assets'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_dividendCover(): #TODO: dividend
    #股息保障倍数＝归属于母公司的净利润/最近 1 年的累计派现金额
    name='Q_dividendCover'
    col1='net_profit_excl_min_int_inc'
    col2='cash_div'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_equityToAsset():
    #股东权益比率=股东权益合计/总资产
    name='Q_equityToAsset'
    col1='tot_shrhldr_eqy_excl_min_int'
    col2='tot_assets'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

'''
是否应该在最后回测的时候才mix各种原子，此处实际上是不同原子的运算组合。实际上是做了
重复的劳动，而且会把后边组合各种因子的过程变得更加无序和复杂。
'''
def get_equityTurnover():
    #股东权益周转率＝营业总收入*2/(期初净资产+期末净资产)
    name='Q_equityTurnover'
    col1='tot_oper_rev'
    col2='tot_assets'
    col3='tot_liab'
    df=get_dataspace([col1,col2,col3])
    df['y']=df[col2]-df[col3]
    df[name]=ratio(df, col1, 'y', smooth=True)
    save_indicator(df,name)

def get_fixedAssetTurnover():
    #营业收入 * 2  /  （期初固定资产 + 期末固定资产）
    name='Q_fixedAssetTrunover'
    col1='tot_oper_rev'
    col2='fix_assets'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2, smooth=True)
    save_indicator(df,name)


def get_grossIncomeRatio():
    #销售毛利率=[营业收入-营业成本]/营业收入
    name='Q_grossIncomeRatio'
    col1='oper_rev'
    col2='oper_cost'
    df=get_dataspace([col1,col2])
    df['x']=df[col1]-df[col2]
    df[name]=ratio(df, 'x', col1)
    save_indicator(df,name)

def get_inventoryTurnover():
    #营业成本 * 2  /  （期初存货净额 + 期末存货净额）
    name='Q_inventoryTrunover'
    col1='oper_rev'
    col2='inventories'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2, smooth=True)
    save_indicator(df,name)

def get_mlev(): #TODO:
    #长期负债/(长期负债+市值)
    '''
    负债=流动负债+非流动负债=短期负债+长期负债
    流动负债=短期负债
    长期负债=非流动负债=长期借款+应付债券+长期应付款
    '''
    name='Q_mlev'
    col1='tot_non_cur_liab'
    col2='freefloat_cap'
    df=get_dataspace([col1,col2])
    df[name]=df[col1]/(df[col1]+df[col2])
    save_indicator(df,name)


def get_netNonOItoTP():
    #营业外收支净额/利润总额
    name='Q_netNonOItoTP'
    col1='non_oper_rev'
    col2='non_oper_exp'
    col3='net_profit_excl_min_int_inc'
    df=get_dataspace([col1,col2,col3])
    df[name]=(df[col1]-df[col2])/df[col3]
    save_indicator(df,name)

def get_netProfitCashCover():
    #经营活动产生的现金流量净额/净利润
    name='Q_netProfitCashCover'
    col1='net_cash_flows_oper_act'
    col2='net_profit_excl_min_int_inc'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_interestCover():
    #TODO:int_exp 数据缺失严重
    #利息保障倍数＝息税前利润/利息费用
    '''
    息税前利润=净利润+所得税+财务费用
    '''
    name='Q_interestCover'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    col4='int_exp'
    df=get_dataspace([col1,col2,col3])
    df[name]=(df[col1]+df[col2]+df[col3])/df[col4]
    save_indicator(df,name)

def get_NPCutToNetRevenue():#TODO:数据缺失
    #扣除非经常损益后的净利润/营业总收入
    name='Q_NPCutToNetRevenue'
    col1='net_profit_after_ded_nr_lp'
    col2='tot_oper_rev'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_netProfitRatio():
    #销售净利率＝含少数股东损益的净利润/营业收入
    name='Q_netProfitRatio'
    col1='net_profit_incl_min_int_inc'
    col2='tot_oper_rev'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_netProfitTorevenue():
    #净利润/营业总收入
    name='Q_netProfitToRevenue'
    col1='net_profit_excl_min_int_inc'
    col2='tot_oper_rev'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_netProfitToTotProfit():
    #净利润/利润总额
    name='Q_netProfitToTotProfit'
    col1='net_profit_excl_min_int_inc'
    col2='oper_profit'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_NPCutToNetProfit():
    #扣除非经常损益后的净利润/归属于母公司的净利润
    name='Q_NPCutToNetProfit'
    col1='net_profit_after_ded_nr_lp' #TODO: 数据缺失严重
    col2='net_profit_excl_min_int_inc'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_operatingExpenseRate():
    #销售费用/营业总收入
    name='Q_operatingExpenseRate'
    col1 = 'selling_dist_exp'
    col2 = 'tot_oper_rev'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_operatingProfitToAsset():
    #营业利润/总资产
    name='Q_operatingProfitToAsset'
    col1='oper_profit'
    col2='tot_assets'
    df = get_dataspace([col1, col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_operatingProfitToEquity():
    #TODO: 所有者权益
    #营业利润/净资产
    name='Q_operatingProfitToEquity'
    col1='oper_profit'
    col2='tot_assets'
    col3='tot_liab'
    df=get_dataspace([col1,col2,col3])
    df[name]=df[col1]/(df[col2]-df[col3])
    save_indicator(df,name)

def get_operCashInToAsset():
    #总资产现金回收率＝经营活动产生的现金流量净额 * 2/(期初总资产+期末总资产)
    name='Q_operCashInToAsset'
    col1='net_cash_flows_oper_act'
    col2='tot_assets'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2, smooth=True)
    save_indicator(df,name)

def get_operCashInToCurrentDebt():
    #现金流动负债比=经营活动产生的现金流量净额/流动负债
    name='Q_operCashInToCurrentDebt'
    col1='net_cash_flows_oper_act'
    col2='oper_rev'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_periodCostsRate():
    #销售期间费用率＝[营业费用+管理费用+财务费用]/营业收入
    name='Q_periodCostsRate'
    col1='oper_cost'
    col2='gerl_admin_exp'
    col3='fin_exp'
    col4='tot_oper_cost'
    df=get_dataspace([col1,col2,col3,col4])
    df[name]=(df[col1]+df[col2]+df[col3])/df[col4]
    save_indicator(df,name)

def get_quickRatio():
    #速动比率＝(流动资产合计-存货)/流动负债合计
    name='Q_quickRatio'
    col1='tot_cur_assets'
    col2='inventories'
    col3='tot_cur_liab'
    df=get_dataspace([col1,col2,col3])
    df[name]=(df[col1]-df[col2])/df[col3]
    save_indicator(df,name)

def get_receivableTopayable():
    #应收应付比 = （应收票据+应收账款） / （应付票据+应付账款）
    name='Q_receivableTopayble'
    col1='notes_rcv'
    col2='acct_rcv'
    col3='notes_payable'
    col4='acct_payable'
    df=get_dataspace([col1,col2,col3,col4])
    df[name]=(df[col1]+df[col2])/(df[col3]+df[col4])
    save_indicator(df,name)

def get_roa():
    #总资产净利率=净利润(含少数股东损益)TTM/总资产
    name='Q_roa'
    col1='net_profit_incl_min_int_inc'
    col2='tot_assets'
    df=get_dataspace([col1,col2])
    df[name]=x_ttm(df[col1])/df[col2]
    save_indicator(df,name)

def get_roe_ebit():
    #总资产报酬率＝息税前利润/总资产
    name='Q_roe_ebit'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    col4='tot_assets'
    df=get_dataspace([col1, col2, col3, col4])
    df[name]=(df[col1]+df[col2]+df[col3])/df[col4]
    save_indicator(df,name)

def get_roa_ebit():
    #归属于母公司的净利润/归属于母公司的股东权益
    name='Q_roe'
    col1='net_profit_excl_min_int_inc'
    col2='tot_shrhldr_eqy_excl_min_int'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_operatingCostToTOR():
    #营业总成本/营业总收入
    name='Q_operatingCostToTOR'
    col1 = 'tot_oper_rev'
    col2 = 'tot_oper_cost'
    df=get_dataspace([col1,col2])
    df[name] = ratio(df, col1, col2)
    save_indicator(df, name)

def get_downturnRisk():
    #std(min(本季度现金流-上一季度现金流,0))
    name='Q_downturnRisk'
    col='net_cash_flows_oper_act'
    df=get_dataspace(col)
    df[name]=x_history_downside_std(df,col)
    save_indicator(df,name)

def get_artRate(): #TODO: use asharefinancialindicator
    # 应收账款周转率
    name='Q_artRate'
    col='s_fa_arturn'
    df=get_dataspace(col)
    save_indicator(df,col)



#------------------2 年标准差--------------------------
def get_earningsStability():
    #净利润过去 2 年的标准差
    name='Q_earningsStability'
    col='net_profit_incl_min_int_inc'
    df=get_dataspace(col)
    df[name]=x_history_std(df[col],q=8)
    save_indicator(df, name)




