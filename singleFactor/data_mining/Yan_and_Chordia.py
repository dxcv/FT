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
import matplotlib.pyplot as plt
from config import DIR_DM, DIR_DM_RESULT,DIR_DM_TMP
from data.dataApi import read_local
from singleFactor.check import check_factor, check_fn, daily_to_monthly
from singleFactor.financial import quarterly_to_daily
from singleFactor.operators import *




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

    bs=pd.read_pickle(os.path.join(DIR_DM_TMP, 'bs.pkl'))
    cf=pd.read_pickle(os.path.join(DIR_DM_TMP, 'cf.pkl'))
    inc=pd.read_pickle(os.path.join(DIR_DM_TMP, 'inc.pkl'))

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

    data.to_pickle(os.path.join(DIR_DM_TMP, 'data.pkl'))

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


data = pd.read_pickle(os.path.join(DIR_DM_TMP, 'data.pkl'))
data = data.set_index(['stkcd', 'report_period'])

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

def cal_and_check(args):
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
        name='2-{}-{}-{}'.format(func,x,y)
        s=eval(func)(data,x,y)
        s.name=name
    result=s.to_frame()
    result['trd_dt']=data['trd_dt']
    daily=quarterly_to_daily(result,name)

    monthly = daily_to_monthly(daily)
    monthly = monthly.stack().to_frame().swaplevel()
    monthly.index.names = ['stkcd', 'month_end']
    monthly = monthly.sort_index()
    monthly.columns = [name]

    directory = os.path.join(DIR_DM_RESULT, name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    daily.to_pickle(os.path.join(directory,'daily.pkl'))
    monthly.to_pickle(os.path.join(directory,'monthly.pkl'))

    try:
        dfs,figs=check_factor(monthly)
        for k in dfs.keys():
            dfs[k].to_csv(os.path.join(directory, k + '.csv'))
        for k in figs.keys():
            figs[k].savefig(os.path.join(directory, k + '.png'))
    except:
        with open(os.path.join(DIR_DM_TMP,'failed.txt'),'a') as f:
            f.write(name+'\n')
        print('{}-------> wrong!'.format(name))

#TODO: 要每期筛选


def get_calculated():
    fns=os.listdir(DIR_DM_RESULT)
    handled=[]
    for fn in fns:
        handled.append(tuple(fn.split('-')[1:]))
    return handled

def run():
    arg_list=get_arg_list()
    print(len(arg_list))
    calculated=get_calculated()
    alist=[arg for arg in arg_list if arg not in calculated]
    print(len(alist))
    pool=multiprocessing.Pool(30)
    pool.map(cal_and_check,alist)


if __name__ == '__main__':
    run()





#TODO: select those indicators with enough sample





