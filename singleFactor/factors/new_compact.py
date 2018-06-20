# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  13:53
# NAME:FT-new_compact.py

from config import SINGLE_D_INDICATOR
from data.dataApi import get_dataspace
import os
from singleFactor.factors.new_operators import x_pct_chg, ratio_yoy_pct_chg, \
    x_history_growth_avg, x_history_compound_growth


def save_indicator(df,name):
    df[['trd_dt',name]].to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))


#---------------------- 1. 单个指标增长率（同比增长率） x_pct_chg(q=4)

# growth
cols=[
    # income
    'oper_rev', # 营业收入
    'tot_oper_rev', # 营业总收入
    # 'tot_oper_cost',
    # 'oper_cost',
    'oper_profit', #营业利润
    'net_profit_excl_min_int_inc', # 净利润
    'net_profit_excl_min_int_inc', #净利润 (不含少数股东权益)
    'net_profit_incl_min_int_inc', #净利润 (含少数股东权益)
    'net_profit_after_ded_nr_lp', # 扣除经常性损益后净利润

    # cash flow
    'net_cash_flows_oper_act',  # 经营现金流

    # balance sheet
    'tot_assets', #总资产

    # others
    's_fa_roe', #roe
    'cash_div', # 股息
]

'''
methods:
    1. x_pct_chg(s,q=4)
    2. x_pct_chg(s,q=12)
    3. x_history_compound_growth(s,q=12)
    4. x_pct_chg(s,q=20)
    5. x_history_compound_growth(s,q=20)
    6. x_history_std(s,q=8) #stability


'''


#--------------------------------compound indicator-----------------------
'''
eps = net_profit_excl_min_int_inc / cap_stk
netOperCashPerShare = net_cash_flows_oper_act / cap_stk
netAsset = tot_assets - tot_liab

methods:
    1. ratio_yoy_pct_chg(df,x,y)
    2. x_pct_chg(s,q=4)

'''

###############################quality##########################################
'''
cashDividendCover = net_cash_flows_oper_act / cash_div
cashRateOfSales = net_cash_flows_oper_act / oper_rev
cashToCurrentLiability = net_cash_flows_oper_act / tot_cur_liab
cashToLiability = net_cash_flows_oper_act / tot_liab
currentAssetToAsset = tot_cur_assets / tot_assets
currentRatio = tot_cur_assets / tot_cur_liab
debtAssetRatio = tot_liab / tot_assets
intanibleAssetRatio = intang_assets / tot_assets
dividendCover = net_profit_excl_min_int_inc / cash_div 股息保障倍数
equityToAsset = tot_shrhldr_eqy_excl_min_int / tot_assets 股东权益比
equityTurnover = tot_oper_rev * 2 / (netAsset + lag(netAsset)) 股东权益周转率＝营业总收入*2/(期初净资产+期末净资产)
fixedAssetTurnover = tot_oper_rev * 2 / (fix_assets + lag(fix_assets)) 营业收入 * 2  /  （期初固定资产 + 期末固定资产）
grossIncomeRatio = (oper_rev - oper_cost) / oper_rev 销售毛利率=[营业收入-营业成本]/营业收入
inventoryTurnover = oper_rev * 2 / (inventories + lag(inventories)) 营业成本 * 2  /  （期初存货净额 + 期末存货净额）
mlev = tot_non_cur_liab / (tot_non_cur_liab + freefloat_cap) 长期负债/(长期负债+市值)
netNonOItoTP = (non_oper_rev - non_oper_exp) / net_profit_excl_min_int_inc 营业外收支净额/利润总额
netProfitCashCover = net_cash_flows_oper_act / net_profit_excl_min_int_inc 
interestCover = (net_profit_excl_min_int_inc + inc_tax + fin_exp + int_exp) / int_exp 利息保障倍数＝息税前利润/利息费用
NPCutToNetRevenue = net_profit_after_ded_nr_lp / tot_oper_rev 扣除非经常损益后的净利润/营业总收入
netProfitRatio = net_profit_incl_min_int_inc / tot_oper_rev  销售净利率＝含少数股东损益的净利润/营业收入
netProfitToRevenue = net_profit_excl_min_int_inc / tot_oper_rev 
netProfitToTotProfit = net_profit_excl_min_int_inc / oper_profit
NPCutToNetProfit =  net_profit_after_ded_nr_lp / net_profit_excl_min_int_inc
operatingExpenseRate = selling_dist_exp / tot_oper_rev
operatingProfitToAsset = oper_profit / tot_assets
operatingProfitToEquity = oper_profit / netAsset
operCashInToAsset = net_cash_flows_oper_act / tot_assets
operCashInToCurrentDebt = net_cash_flows_oper_act / oper_rev 现金流动负债比=经营活动产生的现金流量净额/流动负债
periodCostsRate = (oper_cost + gerl_admin_exp + fin_exp + tot_oper_cost) / tot_oper_cost 销售期间费用率＝[营业费用+管理费用+财务费用]/营业收入
quickRatio = (tot_cur_assets - inventories) / tot_cur_liab 速动比率＝(流动资产合计-存货)/流动负债合计
receivableToPayable = (notes_rcv + acct_rcv) / (notes_payable + acct_payable) 应收应付比 = （应收票据+应收账款） / （应付票据+应付账款）
roa =  x_ttm(net_profit_incl_min_int_inc) / tot_assets 总资产净利率=净利润(含少数股东损益)TTM/总资产
roe_ebit = (net_profit_excl_min_int_inc + inc_tax + fin_exp) / tot_assets 总资产报酬率＝息税前利润/总资产
roe = net_profit_excl_min_int_inc / tot_shrhldr_eqy_excl_min_int
operatingCostToTOR = tot_oper_rev / tot_oper_cost
downturnRisk = x_history_downside_std( net_cash_flows_oper_act ) std(min(本季度现金流-上一季度现金流,0))
artRate = s_fa_arturn 应收账款周转率


methods:
    1. ratio_x_y(df,col1,col2)
    2. x_ttm
    3. x_history_downside_std(s)


'''

####################################value ######################################
'''
bp = tot_shrhldr_eqy_excl_min_int / tot_assets
ep = net_profit_excl_min_int_inc / tot_assets
sp = oper_rev / tot_assets
cfp = net_cash_flows_oper_act / tot_assets
salesEV = oper_rev / (tot_assets + tot_non_cur_liab - monetary_cap)
ebitdaToCap = ebitda / cap
pe = tot_assets / net_profit_excl_min_int_inc (1/EP)n年历史复合增长率
dp = cash_div / tot_assets 
capSquare = x_square (cap) 
ebitToAsset = (net_profit_excl_min_int_inc + inc_tax + fin_exp) / tot_assets



methods:
    1. ratio_x_y(df,col1,col2)
    2. x_history_compound_growth(s)
    3. x_square(s)


'''