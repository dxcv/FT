# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-31  16:06
# NAME:FT-1 generate_indicators.py
import os

# from config import DIR_DM_RESULT, DIR_DM_TMP
from empirical.config_ep import DIR_DM, DIR_DM_INDICATOR
from singleFactor.calculate_indicators.financial import quarterly_to_daily
from singleFactor.old.check import daily_to_monthly
from singleFactor.operators import *
from tools import multi_process

'''
References:
    1. Yan, X. (Sterling), and Zheng, L. (2017). Fundamental Analysis and the Cross-Section of Stock Returns: A Data-Mining Approach. The Review of Financial Studies 30, 1382–1423.

'''
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
                # 'ebit',#fixme: financial does not include this two indicators
                # 'ebitda',
                ]

def get_financial_sheets():
    # bs=read_local_sql('asharebalancesheet',database='ft_zht')
    # cf=read_local_sql('asharecashflow_q',database='ft_zht')
    # inc=read_local_sql('ashareincome_q',database='ft_zht')

    # bs.to_pickle(os.path.join(r'E:\tmp','bs.pkl'))
    # cf.to_pickle(os.path.join(r'E:\tmp','cf.pkl'))
    # inc.to_pickle(os.path.join(r'E:\tmp','inc.pkl'))

    bs=pd.read_pickle(os.path.join(DIR_DM, 'bs.pkl'))
    cf=pd.read_pickle(os.path.join(DIR_DM, 'cf.pkl'))
    inc=pd.read_pickle(os.path.join(DIR_DM, 'inc.pkl'))

    return bs,cf,inc


def combine_financial_sheet():
    bs,cf,inc=get_financial_sheets()

    bs=bs.set_index(['stkcd','report_period'])
    cf=cf.set_index(['stkcd','report_period'])
    inc=inc.set_index(['stkcd','report_period'])

    financial=pd.concat([bs,cf,inc],axis=1)
    financial=financial.reset_index()
    financial=financial[financial['stkcd'].str.slice(0,1).isin(['0','3','6'])]# only keep A share
    financial=financial.sort_values(['stkcd','report_period'])

    #find duplicated columns
    dup_cols=[]
    for col in financial.columns:
        if isinstance(financial[col],pd.DataFrame):
            print(col,financial[col].shape[1])
            if col not in dup_cols:
                dup_cols.append(col)

    #handle duplicated trd_dt
    trd_dt=financial['trd_dt']
    trd_dt_new=trd_dt.apply(max,axis=1)
    del financial['trd_dt']
    financial['trd_dt']=trd_dt_new

    #handle duplicated undistributed_profit
    pr=financial['undistributed_profit']
    pr_new=pr.iloc[:,0]
    del financial['undistributed_profit']
    financial['undistributed_profit']=pr_new

    #handle duplicated unconfirmed_invest_loss
    pr=financial['unconfirmed_invest_loss']
    pr_new=pr.iloc[:,0]
    del financial['unconfirmed_invest_loss']
    financial['unconfirmed_invest_loss']=pr_new

    cols_to_delete=['ann_dt','crncy_code','statement_type','comp_type_code',
                    's_info_compcode','opdate','opmode']
    for col in cols_to_delete:
        del financial[col]
    financial=financial.set_index(['stkcd','report_period'])

    #filter out the indicators with too small sample
    a = financial.notnull().sum()
    cols = a[a > 10000].index
    financial = financial[cols]

    financial.to_pickle(os.path.join(DIR_DM, 'financial.pkl'))


def generator_with_single_variable(func,s):
    return eval(func)(s)

def generator_with_two_variable(func,df,x,y):
    name='2-{}-{}-{}'.format(func,x,y)
    return eval(func)(df,x,y),name

funcs1=['x_chg',
       'x_pct_chg',
       'x_history_growth_avg',
       'x_square',
       # 'x_history_compound_growth', # almost the same as x_history_growth_avg
       'x_history_downside_std',
       # 'x_history_growth_std',
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

funcs3=[
    'ratio3'
]



combine_financial_sheet()

financial = pd.read_pickle(os.path.join(DIR_DM, 'financial.pkl'))

def get_arg_list():
    unuseful_cols = ['stkcd', 'report_period', 'trd_dt']
    variables = [col for col in financial.columns if
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

def get_indicators_monthly(args):
    print(args)
    if len(args)==2:
        func=args[0]
        x=args[1]
        name='1-{}-{}'.format(func,x)
        s=eval(func)(financial[x])
        s.name=name
    else:
        func=args[0]
        x=args[1]
        y=args[2]
        name='2-{}-{}-{}'.format(func,x,y)
        s=eval(func)(financial, x, y)
        s.name=name
    result=s.to_frame()
    result['trd_dt']=financial['trd_dt']
    daily=quarterly_to_daily(result,name)
    # daily.to_pickle(os.path.join())

    monthly = daily_to_monthly(daily)
    if len(monthly)>0:
        monthly = monthly.stack().to_frame().swaplevel()
        monthly.index.names = ['stkcd', 'month_end']
        monthly = monthly.sort_index()
        monthly.columns = [name]

        directory = os.path.join(DIR_DM_INDICATOR, name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # daily.to_pickle(os.path.join(directory,'daily.pkl'))
        monthly.to_pickle(os.path.join(directory,'monthly.pkl'))
# args='ratio_of_growth_rates', 's_fa_eps_basic', 'tot_oper_cost'
# get_indicators_monthly(args)


    # try:
    #     dfs,figs=check_factor(monthly)
    #     for k in dfs.keys():
    #         dfs[k].to_csv(os.path.join(directory, k + '.csv'))
    #     for k in figs.keys():
    #         figs[k].savefig(os.path.join(directory, k + '.png'))
    # except:
    #     with open(os.path.join(DIR_DM_TMP,'failed.txt'),'a') as f:
    #         f.write(name+'\n')
    #     print('{}-------> wrong!'.format(name))

def get_calculated():
    fns=os.listdir(DIR_DM_INDICATOR)
    handled=[]
    for fn in fns:
        handled.append(tuple(fn.split('-')[1:]))
    return handled

def get_all_indicators_monthly():
    arg_list=get_arg_list()
    print(len(arg_list))
    calculated=get_calculated()
    alist=[arg for arg in arg_list if arg not in calculated]
    print(len(alist))

    multi_process(get_indicators_monthly, alist, 10)
    # for al in alist:
    #     get_indicators_monthly(al)
#
# if __name__ == '__main__':
#     get_all_indicators_monthly()


