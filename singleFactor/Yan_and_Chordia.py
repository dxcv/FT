# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-31  16:06
# NAME:FT-Yan_and_Chordia.py

import os
import pandas as pd
from singleFactor.old.base_function import x_pct_chg, ratio_x_y, \
    ratio_yoy_chg, ratio_yoy_pct_chg, ratio_x_chg_over_lag_y, pct_chg_dif, \
    ratio_x_y_history_std, ratio_x_y_history_downside_std, x_history_growth_avg, x_history_growth_std, \
    x_history_growth_downside_std
from singleFactor.old import check

dirtmp=r'e:\tmp'
proj_bi = r'E:\test_yan\indicators\bivariate'
proj_sg = r'e:\test_yan\indicators\single'

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

    cols_to_delete=['ann_dt','crncy_code','statement_type','comp_type_code',
                    's_info_compcode','opdate','opmode']
    for col in cols_to_delete:
        del data[col]

    data.to_pickle(os.path.join(dirtmp,'data.pkl'))


def gen_with_x_y(df, var, base_var):
    # TODO: do not use ttm with q data,how about using ttm method and use accumulative data?
    # x,y
    ind1=ratio_x_y(df, var, base_var, ttm=False, delete_negative_y=True)['target']
    ind1.name='ratio_x_y___{}_{}'.format(var,base_var)

    ind2=ratio_yoy_chg(df, var, base_var, ttm=False, delete_negative_y=True)['target']
    ind2.name='ratio_yoy_chg___{}_{}'.format(var,base_var)

    ind3=ratio_yoy_pct_chg(df, var, base_var, ttm=False, delete_negative_y=True)['target']
    ind3.name='ratio_yoy_pct_chg___{}_{}'.format(var,base_var)

    ind4=ratio_x_chg_over_lag_y(df, var, base_var, ttm=False, delete_negative_y=True)['target']
    ind4.name='ratio_x_chg_over_lag_y___{}_{}'.format(var,base_var)

    ind5=pct_chg_dif(df, var, base_var, ttm=False, delete_negative=True)['target']
    ind5.name='pct_chg_dif___{}_{}'.format(var,base_var)

    #devariate
    ind6=ratio_x_y_history_std(df, var, base_var, q=8, delete_negative_y=True)['target']
    ind6.name='ratio_x_y_history_std___{}_{}'.format(var,base_var)

    ind7=ratio_x_y_history_downside_std(df, var, base_var, q=12, delete_negative_y=True)['target']
    ind7.name='ratio_x_y_history_downside_std___{}_{}'.format(var,base_var)

    return pd.concat([ind1,ind2,ind3,ind4,ind5,ind6,ind7],axis=1)



def gen_with_x(df,var):
    # TODO:base bariable
    # single var :   base_variables + variables
    ind8 = x_pct_chg(df, var, q=1, ttm=False, delete_negative=True)['target']
    ind8.name = 'x_pct_chg___{}'.format(var)

    ind9=x_history_growth_avg(df,var,q=12,ttm=False,delete_negative=True)['target']
    ind9.name='x_history_growth___{}'.format(var)

    ind10=x_history_growth_std(df,var,q=12,delete_negative=True)['target']
    ind10.name='x_history_growth___{}'.format(var)

    ind11=x_history_growth_downside_std(df,var,q=12,delete_negative=True)['target']
    ind11.name='x_history_growth_downside_std___{}'.format(var)

    return pd.concat([ind8,ind9,ind10,ind11],axis=1)

#TODO: add other operator
#TODO: delete indicators with too small sample

#TODO: add ttm at this place

def gen_indicators2(data,base_variables,var):
    directory=os.path.join(proj_bi,var)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for base_var in base_variables:
        bi_ind=gen_with_x_y(data, var, base_var)
        bi_ind.to_csv(os.path.join(directory,base_var+'.csv'))
        print(var,base_var)

def gen_indicators1(data, var):
    sg_ind=gen_with_x(data,var)
    sg_ind.to_csv(os.path.join(proj_sg,var+'.csv'))
    print(var)

def check():
    data = pd.read_pickle(os.path.join(dirtmp, 'data.pkl'))
    data = data.set_index(['stkcd', 'report_period'])
    vars=os.listdir(proj_bi)
    fps=[]
    for var in vars:
        fns=os.listdir(os.path.join(proj_bi,var))
        fps+=[os.path.join(proj_bi,var,fn) for fn in fns]

    fps+=[os.path.join(proj_sg,fn) for fn in os.listdir(proj_sg)]

    for fp in fps:
        df=pd.read_csv(fp,index_col=[0,1],parse_dates=True)
        df['trd_dt']=data['trd_dt']
        for col in [c for c in df.columns if c!='trd_dt']:
            subdf=df[['trd_dt',col]]
            subdf.columns=['trd_dt','target']
            check(subdf,col)
        print(fp)


check()



#TODO: 要每期筛选


# if __name__ == '__main__':
#     data = pd.read_pickle(os.path.join(dirtmp, 'data.pkl'))
#     data = data.set_index(['stkcd', 'report_period'])
#
#     unuseful_cols = ['stkcd', 'report_period', 'trd_dt']
#     variables = [col for col in data.columns if
#                  col not in unuseful_cols + base_variables]
#
#
#     pool=multiprocessing.Pool(4)
#     pool.map(partial(gen_indicators2,data,base_variables),variables)

    # pool=multiprocessing.Pool(4)
    # pool.map(partial(gen_indicators1,data),variables+base_variables)










