# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-31  16:06
# NAME:FT-Yan_and_Chordia.py
import multiprocessing
import os
from functools import partial

import pandas as pd
from singleFactor.check import check_factor
from singleFactor.financial import convert_to_monthly
from singleFactor.operators import *

dir_tmp= r'D:\zht\database\quantDb\internship\FT\singleFactor\data_mining\tmp'
dir_indicators= r'D:\zht\database\quantDb\internship\FT\singleFactor\data_mining\indicators'
dir_check=r'D:\zht\database\quantDb\internship\FT\singleFactor\data_mining\check'

base_variables1=['tot_assets', # total assets
                'tot_cur_assets',# total current assets
                'inventories',#inventory
                '',# property,plant,and equipment
                'tot_liab',#total liabilities
                'tot_cur_liab',# total current liabilities
                'tot_non_cur_liab',# long-term debt
                '' # total common equity = total assets - total liablities
                'tot_shrhldr_eqy_excl_min_int', # stockholders' equity
                'cap_stk',# total invested capital
                'oper_rev',# total sale
                'less_oper_cost', # cost of goods sold
                '',# selling,general,and administrative cost
                '',# number of employees,s_info_totalemployees in ashareintroduction
                '',# market capitalization
                '' # refer to the excel for other indicators as denominator
                ]

base_variables=[
                'tot_assets',
                'tot_cur_assets',
                'inventories',
                'tot_cur_liab',
                'tot_non_cur_liab',
                'tot_liab',
                'cap_stk',
                'tot_shrhldr_eqy_excl_min_int',
                'tot_shrhldr_eqy_incl_min_int',
                'tot_oper_rev',
                'oper_rev',
                'tot_oper_cost',
                'tot_profit',
                'net_profit_incl_min_int_inc',
                'net_profit_excl_min_int_inc',
                'ebit',
                'ebitda',
                ]

def get_financial_sheets():
    # bs=read_local_sql('asharebalancesheet',database='ft_zht')
    # cf=read_local_sql('asharecashflow_q',database='ft_zht')
    # inc=read_local_sql('ashareincome_q',database='ft_zht')

    # bs.to_pickle(os.path.join(r'E:\tmp','bs.pkl'))
    # cf.to_pickle(os.path.join(r'E:\tmp','cf.pkl'))
    # inc.to_pickle(os.path.join(r'E:\tmp','inc.pkl'))

    bs=pd.read_pickle(os.path.join(dir_tmp, 'bs.pkl'))
    cf=pd.read_pickle(os.path.join(dir_tmp, 'cf.pkl'))
    inc=pd.read_pickle(os.path.join(dir_tmp, 'inc.pkl'))

    return bs,cf,inc


def combine_financial_sheet():
    bs,cf,inc=get_financial_sheets()

    bs=bs.set_index(['stkcd','report_period'])
    cf=cf.set_index(['stkcd','report_period'])
    inc=inc.set_index(['stkcd','report_period'])

    data=pd.concat([bs,cf,inc],axis=1)
    data=data.reset_index()
    data=data[data['stkcd'].str.slice(0,1).isin(['0','3','6'])]# only keep A share
    data=data.sort_values(['stkcd','report_period'])

    #find duplicated columns
    dup_cols=[]
    for col in data.columns:
        if isinstance(data[col],pd.DataFrame):
            print(col,data[col].shape[1])
            if col not in dup_cols:
                dup_cols.append(col)

    #handle duplicated trd_dt
    trd_dt=data['trd_dt']
    trd_dt_new=trd_dt.apply(max,axis=1)
    del data['trd_dt']
    data['trd_dt']=trd_dt_new

    #handle duplicated undistributed_profit
    pr=data['undistributed_profit']
    pr_new=pr.iloc[:,0]
    del data['undistributed_profit']
    data['undistributed_profit']=pr_new

    #handle duplicated unconfirmed_invest_loss
    pr=data['unconfirmed_invest_loss']
    pr_new=pr.iloc[:,0]
    del data['unconfirmed_invest_loss']
    data['unconfirmed_invest_loss']=pr_new

    cols_to_delete=['ann_dt','crncy_code','statement_type','comp_type_code',
                    's_info_compcode','opdate','opmode']
    for col in cols_to_delete:
        del data[col]

    data.to_pickle(os.path.join(dir_tmp, 'data.pkl'))

def generator_with_single_variable(func,s):
    return eval(func)(s)

def generator_with_two_variable(func,df,x,y):
    name='2-{}-{}-{}'.format(func,x,y)
    return eval(func)(df,x,y),name

funcs1=['x_chg',
       'x_pct_chg',
       'x_history_growth_avg',
       'x_square',
       'x_history_compound_growth',
       'x_history_downside_std',
       'x_history_growth_std',
       'x_history_growth_downside_std',
       ]

funcs2=[
    'ratio',
    'ratio_chg',
    'ratio_pct_chg',
    'ratio_history_std',
    'ratio_history_compound_growth',
    'ratio_history_downside_std',
    'pct_chg_dif',
    'ratio_x_chg_over_lag_y',
    'ratio_of_growth_rates',
]



data = pd.read_pickle(os.path.join(dir_tmp, 'data.pkl'))
data = data.set_index(['stkcd', 'report_period'])

def _save(df):
    name=df.columns[0]
    df.to_pickle(os.path.join(dir_indicators, name + '.pkl'))

def get_arg_list():
    unuseful_cols = ['stkcd', 'report_period', 'trd_dt']
    variables = [col for col in data.columns if
                 col not in unuseful_cols + base_variables]

    arg_list=[]

    for func in funcs1:
        for var in variables+base_variables:
            arg_list.append((func,var))

    for func in funcs2:
        for y in base_variables:
            for x in variables:
                arg_list.append((func,x,y))
    return arg_list

def cal_indicator(args):
    print(args)
    if len(args)==2:
        func=args[0]
        x=args[1]
        name='1-{}-{}'.format(func,x)
        s=eval(func)(data[x])
        s.name=name
    else:
        func=args[0]
        x=args[1]
        y=args[2]
        name='1-{}-{}-{}'.format(func,x,y)
        s=eval(func)(data,x,y)
        s.name=name
    result=s.to_frame()
    result['trd_dt']=data['trd_dt']
    result=convert_to_monthly(result)[[name]]
    _save(result)

def get_indicators():
    arg_list=get_arg_list()
    pool=multiprocessing.Pool(4)
    pool.map(cal_indicator,arg_list)
    # for i,args in enumerate(arg_list[:100]):
    #     cal_indicator(args)
    #     print(i,args)


#TODO: 要每期筛选

def _check_a_indicator(fn):
    print(fn)
    df=pd.read_pickle(os.path.join(dir_indicators,fn))
    try:
        check_factor(df,rootdir=dir_check)
    except:
        print('{}-------> wrong!'.format(fn))


def check_indicators():
    fns=os.listdir(dir_indicators)
    pool=multiprocessing.Pool(4)
    pool.map(_check_a_indicator,fns)

def debug():
    fn=r'1-x_chg-acting_trading_sec.pkl'
    df=pd.read_pickle(os.path.join(dir_indicators,fn))
    check_factor(df,rootdir=dir_check)


if __name__ == '__main__':
    get_indicators()



#TODO: select those indicators with enough sample










