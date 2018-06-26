# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  13:53
# NAME:FT-new_compact.py


#---------------------- 1. 单个指标增长率（同比增长率） x_pct_chg(q=4)

# growth
cols=[
    # income
    'oper_rev', # 营业收入
    'tot_oper_rev', # 营业总收入
    # 'tot_oper_cost',
    # 'oper_cost',
    'oper_profit', #营业利润
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
    3. x_pct_chg(s,q=20)
    4. x_history_compound_growth(s,q=12)
    5. x_history_compound_growth(s,q=20)
    6. x_history_std(s,q=8) #stability


'''


#--------------------------------compound indicator-----------------------
'''
eps = net_profit_excl_min_int_inc / cap_stk
netOperCashPerShare = net_cash_flows_oper_act / cap_stk
netAsset = tot_assets - tot_liab

methods:
    1. ratio_pct_chg(df,x,y)
    2. x_pct_chg(s,q=4)

'''

###############################quality##########################################
'''




downturnRisk = x_history_downside_std( net_cash_flows_oper_act ) std(min(本季度现金流-上一季度现金流,0))
artRate = s_fa_arturn 应收账款周转率


methods:
    1. ratio(df,col1,col2)
    2. x_ttm
    3. x_history_downside_std(s)


'''

####################################value ######################################
'''

capSquare = x_square (cap) 


methods:
    1. ratio(df,col1,col2)
    2. x_history_compound_growth(s)
    3. x_square(s)


'''








