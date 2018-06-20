# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  13:32
# NAME:FT-new_value.py
import os

from config import SINGLE_D_INDICATOR
from data.dataApi import get_dataspace
from singleFactor.factors.new_operators import ratio_x_y, \
    x_history_compound_growth, x_square


def save_indicator(df,name):
    df[['trd_dt',name]].to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))




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
