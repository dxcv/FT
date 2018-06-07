# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-31  16:06
# NAME:FT-Yan.py
from functools import reduce

from data.dataApi import read_local_sql
import os
import pandas as pd

dirtmp=r'e:\tmp'

def get_financial_sheets():
    # bs=read_local_sql('asharebalancesheet',database='ft_zht')
    # cf=read_local_sql('asharecashflow_q',database='ft_zht')
    # inc=read_local_sql('ashareincome_q',database='ft_zht')

    # bs.to_pickle(os.path.join(r'E:\tmp','bs.pkl'))
    # cf.to_pickle(os.path.join(r'E:\tmp','cf.pkl'))
    # inc.to_pickle(os.path.join(r'E:\tmp','inc.pkl'))

    bs=pd.read_pickle(os.path.join(dirtmp,'bs.pkl'))
    cf=pd.read_pickle(os.path.join(dirtmp,'cf.pkl'))
    inc=pd.read_pickle(os.path.join(dirtmp,'inc.pkl'))

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

    data.to_pickle(os.path.join(dirtmp,'data.pkl'))

data=pd.read_pickle(os.path.join(dirtmp,'data.pkl'))



base_variables=['tot_assets', # total assets
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





