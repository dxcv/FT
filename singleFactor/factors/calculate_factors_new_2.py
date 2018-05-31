# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-31  13:02
# NAME:FT-calculate_factors_new_2.py
import multiprocessing
import os
import pickle

import pandas as pd
from config import DCC
from data.dataApi import read_local
from singleFactor.factors.base_function import raw_level, x_history_std, \
    x_pct_chg, x_history_compound_growth, ratio_x_y, ratio_yoy_pct_chg, \
    x_history_growth_avg, ttm_adjust, x_history_downside_std
from singleFactor.factors.check import check_factor


def check_raw_level(df,col,name):
    r_ttm=raw_level(df, col, ttm=True)
    r=raw_level(df, col, ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,'{}'.format(name))

def check_stability(df,col,name,q=8):
    r_ttm=x_history_std(df,col,q=q,ttm=True)
    r=x_history_std(df,col,q=q,ttm=False)
    check_factor(r_ttm,name+'_ttm')
    check_factor(r,name)

def check_g_yoy(df, col, name,q=4):
    '''
    yoy 增长率
    Args:
        tbname:
        col: 要检验的指标
        name: 保存的文件夹名
        q:int,q=4 表示yoy
    '''
    r_ttm=x_pct_chg(df,col,q=q,ttm=True)
    r=x_pct_chg(df,col,q=q,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,name)

def check_compound_g_yoy(df,col,name,q=20):
    '''
    复合增长率
    '''
    r_ttm=x_history_compound_growth(df, col, q=q, ttm=True)
    r=x_history_compound_growth(df, col, q=q, ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,name)

def check_ratio(df,colx,coly,name):
    '''x/y'''
    ratio=ratio_x_y(df,colx,coly,ttm=False)
    check_factor(ratio,name)

def check_ratio_yoy_pct_chg(df,colx,coly,name):
    '''
    yoy growth rate of x/y
    '''
    r_ttm=ratio_yoy_pct_chg(df,colx,coly,ttm=True)
    r=ratio_yoy_pct_chg(df,colx,coly,ttm=False)
    check_factor(r_ttm,'{}_ttm'.format(name))
    check_factor(r,name)

#===============================indicator_api===================================
def get_fields_map():
    tbnames=[
    'equity_selected_balance_sheet',
    # 'equity_selected_cashflow_sheet',
    'equity_selected_cashflow_sheet_q',
    # 'equity_selected_income_sheet',
    'equity_selected_income_sheet_q',
    ]

    shared_cols=['stkcd','trd_dt','ann_dt','report_period']
    fields_map={}
    for tbname in tbnames:
        df=read_local(tbname)
        indicators=[col for col in df.columns if col not in shared_cols]
        for ind in indicators:
            if ind not in fields_map.keys():
                fields_map[ind]=tbname
            else:
                raise ValueError('Different tables share the indicator -> "{}"'.format(ind))
    #TODO: cache for fields_map

    with open(os.path.join(DCC,'fields_map.pkl'),'wb') as output:
        pickle.dump(fields_map,output)

# get_fields_map()

def get_dataspace(fields):
    if isinstance(fields,str): #only one field
        fields=[fields]
    with open(os.path.join(DCC,'fields_map.pkl'),'rb') as f:
        fields_map=pickle.load(f)
    dfnames=list(set([fields_map[f] for f in fields]))
    if len(dfnames)==1:
        df=read_local(dfnames[0])
    else:
        df=pd.concat([read_local(dn) for dn in dfnames])
    return df

#=============================single field======================================
'''
factor is based on a single field
'''
def with_single_field(field,func,name):
    df=get_dataspace(field)
    func(df,field,name)
#===============================成长因子========================================
def get_saleEarnings_sq_yoy():
    # 单季度营业利润同比增长率 saleEarnings_sq_yoy
    name='saleEarnings_sq_yoy'
    df=get_dataspace('oper_profit')
    check_g_yoy(df,'oper_profit',name)
'''
def get_saleEarnings_sq_yoy():
    # 单季度营业利润同比增长率 saleEarnings_sq_yoy
    field='oper_profit'
    func=check_g_yoy
    name='saleEarnings_sq_yoy'
    with_single_field(field,func,name)
'''
def get_netProfit3YRAvg():
    #3 年净利润增长率的平均值
    name='netProfit3YRAvg'
    df=get_dataspace('net_profit_excl_min_int_inc')
    df['result']=x_history_growth_avg(df,'net_profit_excl_min_int_inc',q=12)
    check_raw_level(df,'result',name)

def get_earnings_sq_yoy():
    #单季度净利润同比增长率 earnings_sq_yoy
    name='earnings_sq_yoy'
    col = 'net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    check_g_yoy(df, col, name)

def get_sales_sq_yoy():
    #单季度营业收入同比增长率 sales_sq_yoy
    name='sales_sq_yoy'
    col='oper_rev'
    df=get_dataspace(col)
    check_g_yoy(df, col, name)

def get_eps1Ygrowth_yoy():
    #每股收益同比增长率 eps1Ygrowth_yoy
    name='esp1Ygrowth_yoy'
    df=get_dataspace(['net_profit_excl_min_int_inc','cap_stk'])
    check_ratio_yoy_pct_chg(df,'net_profit_excl_min_int_inc','cap_stk',name)

def get_ocfGrowth_yoy():
    #经营现金流增长率 ocfGrowth_yoy
    name='ocfGrowth_yoy'
    col='net_cash_flows_oper_act'
    df=get_dataspace(col)
    check_g_yoy(df,col,name)

def get_earnings_ltg():
    #净利润过去 5 年历史增长率 earnings_ltg
    name='earnings_ltg'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    check_g_yoy(df,col,name,q=20)

def get_sales_ltg():
    # 营业收入过去 5 年历史增长率 sales_ltg
    name='sales_ltg'
    col = 'oper_rev'
    df=get_dataspace(col)
    check_g_yoy(df,col,name,q=20)

def get_g_totalOperatingRevenue():
    #营业总收入增长率
    name='g_totalOperatingRevenue'
    col='tot_oper_rev'
    df=get_dataspace(col)
    check_g_yoy(df,col,name)

def get_g_operatingRevenueCAGR3():
    #营业收入 3 年复合增长率
    name='g_operatingRevenueCAGR3'
    df=get_dataspace('oper_rev')
    df['result']=x_history_compound_growth(df, 'oper_rev', q=12)
    check_raw_level(df,'result',name)

def get_g_operatingRevenueCAGR5():
    #营业收入 5 年复合增长率
    name='g_operatingRevenueCAGR5'
    df=get_dataspace('oper_rev')
    df['result']=x_history_compound_growth(df, 'oper_rev', q=20)
    check_raw_level(df,'result',name)

def get_g_netCashFlow():
    #净现金流增长率 g_netCashFlow
    name='g_netCashFlow'
    col = 'net_cash_flows_oper_act'
    df=get_dataspace(col)
    check_g_yoy(df,col,name)

def get_g_netProfit12Qavg():
    #过去 12 个季度净利润平均年增长率 g_netProfit12Qavg
    name='g_netProfit12Qavg'
    col = 'net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df['result']=x_history_growth_avg(df,col,q=12)
    check_raw_level(df,'result',name)

#TODO:define a standard function for this type of indicators
def get_g_totalOperatingRevenue12Qavg():
    #过去 12 个季度营业总收入平均年增长率
    name='g_totalOperatingRevenue12Qavg'
    col='tot_oper_rev'
    df=get_dataspace(col)
    df['result']=x_history_growth_avg(df,col,q=12)
    check_raw_level(df,'result',name)

def get_g_totalAssets():
    #总资产增长率
    name='g_totalAssets'
    col='tot_assets'
    df=get_dataspace(col)
    check_g_yoy(df,col,name)

def get_g_epscagr5():
    #EPS 5年复合增长率
    name='g_epscagr5'
    col='oper_profit'
    df=get_dataspace(col)
    check_compound_g_yoy(df,col,name,q=20)

def get_netOperateCashFlowPerShare():
    #每股经营活动净现金流增长率
    name='g_netOperateCashFlowPerShare'
    colx='net_cash_flows_oper_act'
    coly='cap_stk'
    df=get_dataspace([colx,coly])
    check_ratio_yoy_pct_chg(df,colx,coly,name)

def get_roe_growth_rate():
    # ROE 增长率
    name='g_roe'
    col='s_fa_roe'
    df=get_dataspace(col)
    check_g_yoy(df,col,name)

def get_dividend3YR():
    #股息3年复合增长率
    name='g_dividend3YR'
    df=get_dataspace('cash_div')
    df['result']=x_history_compound_growth(df, 'cash_div', q=12, ttm=False)
    check_raw_level(df,'result',name)

def get_g_NetProfit():
    #净利润增长率
    name='g_NetProfit'
    df=get_dataspace('net_profit_excl_min_int_inc')
    check_g_yoy(df,'net_profit_excl_min_int_inc',name)

def get_NetProfitCAGR3():
    #净利润 3 年复合增长率
    name='NetProfitCAGR3'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df['result']=x_history_compound_growth(df, col, q=12)
    check_raw_level(df,'result',name)

def get_NetProfitCAGR5():
    #净利润 5 年复合增长率
    name='NetProfitCAGR5'
    col='net_profit_excl_min_int_inc'
    df=get_dataspace(col)
    df['result']=x_history_compound_growth(df, col, q=20)
    check_raw_level(df,'result',name)

def get_g_netAssetsPerShare():
    #每股净资产增长率
    name='g_netAssetsPerShare'
    df=get_dataspace(['tot_assets','tot_liab'])
    df['net_asset']=df['tot_assets']-df['tot_liab']
    check_g_yoy(df,'net_asset',name)

#===================================质量因子====================================
def get_artRate(): #TODO:
    # 应收账款周转率
    name='artRate'
    tbname = 'asharefinancialindicator'
    col='s_fa_arturn'
    check_raw_level(tbname,col,name)

def get_cashRateOfSales():
    # 经营活动产生的现金流量净额/营业收入
    name='cashRateOfSales'
    cols=['net_cash_flows_oper_act','oper_rev']
    df=get_dataspace(cols)
    check_ratio(df,cols[0],cols[1],name)

#TODO: 对于累计的指标，不能使用ttm ,检查那些来自累计报表的指标
def get_cashToCurrentLiability():
    #经营活动产生现金流量净额/流动负债
    name='cashToCurrentLiability'
    cols=['net_cash_flows_oper_act','tot_cur_liab']
    df=get_dataspace(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_cashToLiability():
    #经营活动产生现金流量净额/负债合计
    name='cashToLiability'
    cols=['net_cash_flows_oper_act','tot_liab']
    df=get_dataspace(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_currentAssetsToAsset():
    #流动资产/总资产
    name='currentAssetsToAsset'
    cols=['tot_cur_assets','tot_assets']
    df=get_dataspace(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_currentRatio():
    #流动资产/流动负债
    name='currentRatio'
    cols=['tot_cur_assets','tot_cur_liab']
    df=get_dataspace(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_debtAssetsRatio():
    #总负债/总资产
    name='debtAssetsRatio'
    cols=['tot_liab','tot_assets']
    df=get_dataspace(cols)
    check_ratio(df,cols[0],cols[1],name)

def get_earningsStability():
    #净利润过去 2 年的标准差
    name='earningsStability'
    col='net_profit_incl_min_int_inc'
    df=get_dataspace(col)
    check_stability(df,col,name)

def get_intanibleAssetRatio():
    #无形资产/总资产
    name='intangibleAssetRatio'
    cols=['intang_assets','tot_assets']
    df=get_dataspace(cols)
    check_ratio(df,cols[0],cols[1],name)


def get_interestCover():
    #利息保障倍数＝息税前利润/利息费用
    '''
    息税前利润=净利润+所得税+财务费用'''
    name='interestCover'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    df=get_dataspace([col1,col2,col3])
    df['x']=df[col1]+df[col2]+df[col3]
    df['result']=df['x']/df['int_exp']
    check_raw_level(df,'result',name)

def get_netProfitTorevenue():
    #净利润/营业总收入
    name='netProfitToRevenue'
    colx='net_profit_excl_min_int_inc'
    coly='tot_oper_rev'
    df=get_dataspace([colx,coly])
    check_ratio(df,colx,coly,name)

def get_netProfitToTotProfit():
    #净利润/利润总额
    name='netProfitToTotProfit'
    colx='net_profit_excl_min_int_inc'
    coly='oper_profit'
    df=get_dataspace([colx,coly])
    check_ratio(df,colx,coly,name)

def get_NPCutToNetProfit():
    #扣除非经常损益后的净利润/归属于母公司的净利润
    name='NPCutToNetProfit'
    colx='net_profit_after_ded_nr_lp'
    coly='net_profit_excl_min_int_inc'
    df=get_dataspace([colx,coly])
    check_ratio(df,colx,coly,name)

def get_operatingExpenseRate():
    #销售费用/营业总收入
    name='operatingExpenseRate'
    colx = 'selling_dist_exp'
    coly = 'tot_oper_rev'
    df=get_dataspace([colx,coly])
    check_ratio(df,colx,coly,name)

def get_operatingCostToTOR():
    #营业总成本/营业总收入
    name='operatingCostToTOR'
    colx = 'tot_oper_rev'
    coly = 'tot_oper_cost'
    df=get_dataspace([colx,coly])
    check_ratio(df,colx,coly,name)

def get_operatingProfitToAsset():
    #营业利润/总资产
    name='operatingProfitToAsset'
    colx='oper_profit'
    coly='tot_assets'
    df=get_dataspace([colx,coly])
    check_ratio(df,colx,coly,name)

def get_operatingProfitToEquity():
    #营业利润/净资产
    name='operatingProfitToEquity'
    df=get_dataspace(['oper_profit','tot_assets','tot_liab'])
    df['result']=df['oper_profit']/(df['tot_assets']-df['tot_liab'])
    check_raw_level(df,'result',name)

def get_operCashInToAsset():
    #总资产现金回收率＝经营活动产生的现金流量净额 * 2/(期初总资产+期末总资产)
    name='operCashInToAsset'
    df=get_dataspace(['net_cash_flows_oper_act','tot_assetss'])
    df['x']=df['net_cash_flows_oper_act']*2
    df['y']=df['tot_assets'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['result']=df['x']/df['y']
    check_raw_level(df,'result',name)

def get_operCashInToCurrentDebt():
    #现金流动负债比=经营活动产生的现金流量净额/流动负债
    name='operCashInToCurrentDebt'
    df=get_dataspace(['net_cash_flows_oper_act','oper_rev'])
    df['result']=df['net_cash_flows_oper_act']/df['oper_rev']
    check_raw_level(df,'result',name)

def get_quickRatio():
    #速动比率＝(流动资产合计-存货)/流动负债合计
    name='quickRatio'
    df=get_dataspace(['tot_cur_assets','inventories','tot_cur_liab'])
    df['result']=(df['tot_cur_assets']-df['inventories'])/df['tot_cur_liab']
    check_raw_level(df,'result',name)

def get_receivableTopayable():
    #应收应付比 = （应收票据+应收账款） / （应付票据+应付账款）
    name='receivableTopayble'
    df=get_dataspace(['notes_rcv','acct_rcv','notes_payable','acct_payable'])
    df['result']=(df['notes_rcv']+df['acct_rcv'])/(df['notes_payable']+df['acct_payable'])
    check_raw_level(df,'result',name)

def get_roa():
    #总资产净利率=净利润(含少数股东损益)TTM/总资产
    name='roa'
    df=get_dataspace(['net_profit_incl_min_int_inc'])
    df['ttm']=ttm_adjust(df['net_profit_incl_min_int_inc'])
    df['result']=df['ttm']/df['tot_assets']
    check_raw_level(df,'result',name)

def get_roe_ebit():
    #总资产报酬率＝息税前利润/总资产
    name='roe_ebit'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    df=get_dataspace([col1,col2,col3])
    df['result']=(df[col1]+df[col2]+df[col3])/df['tot_assets']
    check_raw_level(df,'result',name)

def get_roa_ebit():
    #归属于母公司的净利润/归属于母公司的股东权益
    name='roa_ebit'
    colx='net_profit_excl_min_int_inc'
    coly='tot_shrhldr_eqy_excl_min_int'
    df=get_dataspace([colx,coly])
    df['result']=df[colx]/df[coly]
    check_raw_level(df,'result',name)

def get_cashDividendCover():
    #现金股利保障倍数＝经营活动产生的现金流量净额/累计合计派现金额
    name='cashDividendCover'
    df=get_dataspace(['net_cash_flow_oper_act','cash_div'])
    df['result']=df['net_cash_flow_oper_act']/df['cash_div']
    check_raw_level(df,'result',name)

def get_dividendCover():
    #股息保障倍数＝归属于母公司的净利润/最近 1 年的累计派现金额
    name='dividendCover'
    df=get_dataspace(['net_profit_excl_min_int_inc','cash_div'])
    df['result'] = df['net_profit_excl_min_int_inc'] / df['cash_div']
    check_raw_level(df, 'result', name)

def get_ebitToTLiablity():
    #总资产报酬率＝息税前利润/总资产
    name='ebitToTLiability'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    col4='tot_oper_rev'
    df=get_dataspace([col1,col2,col3])
    df['result']=(df[col1]+df[col2]+df[col3])/df[col4]
    check_raw_level(df,'result',name)

def get_equityToAsset():
    #股东权益合计/总资产
    name='equityToAsset'
    colx='tot_shrhldr_eqy_excl_min_int'
    coly='tot_assets'
    df=get_dataspace([colx,coly])
    check_ratio(df,colx,coly,name)

def get_equityTurnover():
    #股东权益周转率＝营业总收入*2/(期初净资产+期末净资产)
    name='equityTurnover'
    df=get_dataspace(['tot_oper_rev','tot_assets','tot_liab'])
    df['x']=df['tot_oper_rev']*2
    df['net_asset']=df['tot_assets']-df['tot_liab']
    df['y']=df['net_assets'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['result']=df['x']/df['y']
    check_raw_level(df,'result',name)

def get_fixedAssetTurnover():
    #营业收入 * 2  /  （期初固定资产 + 期末固定资产）
    name='fixedAssetTrunover'
    df=get_dataspace(['tot_oper_rev','fix_assets'])
    df['x']=df['tot_oper_rev']*2
    df['y']=df['fix_assets'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['result']=df['x']/df['y']
    check_raw_level(df,'result',name)

def get_grossIncomeRatio():
    #销售毛利率=[营业收入-营业成本]/营业收入
    name='grossIncomeRatio'
    df=get_dataspace(['oper_rev','oper_cost'])
    df['result']=(df['oper_rev']-df['oper_cost'])/df['oper_rev']
    check_raw_level(df,'result',name)

def get_inventoryTurnover():
    #营业成本 * 2  /  （期初存货净额 + 期末存货净额）
    name='inventoryTrunover'
    df=get_dataspace(['oper_rev','inventories'])
    df['x']=df['oper_rev']*2
    df['y']=df['inventories'].groupby('stkcd').apply(lambda s:s+s.shift(1))
    df['result']=df['x']/df['y']
    check_raw_level(df,'result',name)

def get_mlev(): #TODO:
    #长期负债/(长期负债+市值)
    '''
    负债=流动负债+非流动负债=短期负债+长期负债
    流动负债=短期负债
    长期负债=非流动负债=长期借款+应付债券+长期应付款
    '''
    name='mlev'
    df1=read_local('equity_fundamental_info')
    df2=read_local('equity_selected_balance_sheet')
    df2=df2.reset_index().sort_values(['stkcd','trd_dt','report_period'])
    df2=df2[~df2.duplicated(subset=['stkcd','trd_dt'],keep='last')].set_index(['stkcd','trd_dt'])
    df=pd.concat([df2,df1],join='inner',axis=1).reset_index()
    df=df.set_index(['stkcd','report_period']).sort_index()
    df['result']=df['tot_non_cur_liab']/(df['tot_non_cur_liab']+df['freefloat_cap'])
    check_raw_level(df,'result',name)

def get_netNonOItoTP():
    #营业外收支净额/利润总额
    name='netNonOItoTP'
    df=get_dataspace(['non_oper_rev','non_oper_exp','net_profit_excl_min_int_inc'])
    df['result']=(df['non_oper_rev']-df['non_oper_exp'])/df['net_profit_excl_min_int_inc']
    check_raw_level(df,'result',name)

def get_netProfitCashCover():
    #经营活动产生的现金流量净额/净利润
    name='netProfitCashCover'
    df=get_dataspace(['net_cash_flows_oper_act','net_profit_excl_min_int_inc'])
    df['result']=df['net_cash_flows_oper_act']/df['net_profit_excl_min_int_inc']
    check_raw_level(df,'reuslt',name)

def get_NPCutToNetRevenue():
    #扣除非经常损益后的净利润/营业总收入
    name='NPCutToNetRevenue'
    df=get_dataspace(['net_profit_after_ded_nr_lp','tot_oper_rev'])
    df['result']=df['net_profit_after_ded_nr_lp']/df['tot_oper_rev']
    check_raw_level(df,'result',name)

def get_netProfitRatio():
    #销售净利率＝含少数股东损益的净利润/营业收入
    name='netProfitRatio'
    df=get_dataspace(['net_profit_incl_min_int_inc','tot_oper_rev'])
    df['result']=df['net_profit_incl_min_int_inc']/df['tot_oper_rev']
    check_raw_level(df,'result',name)

def get_periodCostsRate():
    #销售期间费用率＝[营业费用+管理费用+财务费用]/营业收入
    name='periodCostsRate'
    df=get_dataspace(['oper_cost','gerl_admin_exp','fin_exp','tot_oper_cost'])
    df['result']=(df['oper_cost']+df['gerl_admin_exp']+df['fin_exp'])/df['tot_oper_cost']
    check_raw_level(df,'result',name)

def get_downturnRisk():
    #std(min(本季度现金流-上一季度现金流,0))
    name='downturnRisk'
    df=get_dataspace('net_cash_flows_oper_act')
    df['result']=x_history_downside_std(df,'net_cash_flows_oper_act',q=8)
    check_raw_level(df,'result',name)

def task(f):
    eval(f)()
if __name__=='__main__':
    fstrs=[f for f in locals().keys() if (f.startswith('get') and f!='get_ipython')]
    pool=multiprocessing.Pool(4)
    pool.map(task,fstrs)


#TODO: repalace the long table name with compact name
#TODO: operator (col1+col2-col3)*2/(col1+col1(-1))
#TODO: std(8,col1)
