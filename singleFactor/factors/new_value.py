# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  13:32
# NAME:FT-new_value.py
import os

from config import SINGLE_D_INDICATOR_FINANCIAL
from data.dataApi import get_dataspace
from singleFactor.factors.new_operators import ratio, \
    x_history_compound_growth, x_square


def save_indicator(df,name):
    df[['trd_dt',name]].to_pickle(os.path.join(SINGLE_D_INDICATOR_FINANCIAL, name + '.pkl'))

def get_bp():
    #股东权益合计  /  总市值
    name='V_bp'
    col1='tot_shrhldr_eqy_excl_min_int'
    col2='tot_assets'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_ep():
    #净利润(不含少数股东损益)  /  总市值
    name='V_ep'
    col1='net_profit_excl_min_int_inc'
    col2='tot_assets'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_sp():
    #营业收入  /  总市值
    name='V_sp'
    col1='oper_rev'
    col2='tot_assets'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_cfp():
    #经营活动产生的现金流量净额  /  总市值
    name='V_cfp'
    col1='net_cash_flows_oper_act'
    col2='tot_assets'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_sales_EV():
    #营业收入/(总市值+非流动负债合计-货币资金)
    name='V_sales_EV'
    col1='oper_rev'
    col2='tot_assets'
    col3='tot_non_cur_liab'
    col4='monetary_cap'
    df=get_dataspace([col1,col2,col3,col4])
    df[name]=df[col1]/(df[col2]+df[col3]-df[col4])
    save_indicator(df,name)

def get_ebitda_p():
    #息税折旧摊销前利润  /  总市值
    name='V_ebitda_p'
    col1='ebitda'
    col2='cap'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_peg_nY():
    #(1/EP)n年历史复合增长率
    name='V_peg_nY'
    col1 = 'tot_assets'
    col2 = 'net_profit_excl_min_int_inc'
    df=get_dataspace([col1,col2])
    df['tmp']=df[col1]/df[col2]
    df[name]=x_history_compound_growth(df['tmp'])
    save_indicator(df,name)

def get_dp_new():
    name='V_dp'
    col1='cash_div'
    col2='tot_assets'
    df=get_dataspace([col1,col2])
    df[name]=ratio(df, col1, col2)
    save_indicator(df,name)

def get_p_square():#TODO: add cap to dataspace,compare with get_mlev
    #市值平方
    name='V_p_square'
    col='cap'
    df=get_dataspace(col)
    df[name]=x_square(df[col])
    save_indicator(df,name)

def get_ebitToTLiablity():
    #总资产报酬率＝息税前利润/总资产
    name='G_ebitToTLiability'
    col1='net_profit_excl_min_int_inc'
    col2='inc_tax'
    col3='fin_exp'
    col4='tot_oper_rev'
    df=get_dataspace([col1,col2,col3,col4])
    df[name]=(df[col1]+df[col2]+df[col3])/df[col4]
    save_indicator(df,name)



#TODO: 统计每个截面样本量
#TODO: send a freq parameter to determine which frequency to use.
#TODO: 对于累计的指标，不能使用ttm ,检查那些来自累计报表的指标
