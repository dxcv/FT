# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-04  09:29
# NAME:FT-cal_tools.py


import os
import pickle
import pandas as pd

from config import DCC
from data.dataApi import read_local_pkl
from singleFactor.factors.base_function import raw_level, x_history_std, \
    x_pct_chg, x_history_compound_growth, ratio_x_y, ratio_yoy_pct_chg, \
    raw_square
from singleFactor.factors.check import check_factor


def check_raw_level(df,col,name):
    df_ttm=raw_level(df, col,ttm=True)
    df=raw_level(df, col,ttm=False)
    check_factor(df_ttm,'{}_ttm'.format(name))
    check_factor(df,name)

def check_level_square(df,col,name):
    df_ttm=raw_square(df,col,ttm=True)
    df=raw_square(df,col,ttm=False)
    check_factor(df_ttm,'{}_ttm'.format(name))
    check_factor(df,name)

def check_stability(df,col,name,q=8):
    r_ttm=x_history_std(df,col,q=q,ttm=True)
    r=x_history_std(df,col,q=q,ttm=False)
    check_factor(r_ttm,name+'_ttm')
    check_factor(r,name)

def check_g_yoy(df, col, name,q=4):
    '''
    yoy 增长率
    Args:
        df:
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
def _get_fields_map():
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
        df=read_local_pkl(tbname)
        indicators=[col for col in df.columns if col not in shared_cols]
        for ind in indicators:
            if ind not in fields_map.keys():
                fields_map[ind]=tbname
            else:
                raise ValueError('Different tables share the indicator -> "{}"'.format(ind))
    #TODO: cache for fields_map

    return fields_map

def read_fields_map(refresh=False):
    path=os.path.join(DCC,'fields_map.pkl')
    if not os.path.exists(path):
        fields_map=_get_fields_map()
        with open(path,'wb') as f:
            pickle.dump(fields_map,f)
    elif refresh:
        fields_map = _get_fields_map()
        with open(path, 'wb') as f:
            pickle.dump(fields_map, f)
    else:
        with open(os.path.join(DCC,'fields_map.pkl'),'rb') as f:
            fields_map=pickle.load(f)

    return fields_map


def get_dataspace(fields):
    fields_map=read_fields_map()
    if isinstance(fields,str): #only one field
        fields=[fields]


    dfnames=list(set([fields_map[f] for f in fields]))
    if len(dfnames)==1:
        df=read_local_pkl(dfnames[0])
    else:
        df=pd.concat([read_local_pkl(dn) for dn in dfnames], axis=1)
    return df
