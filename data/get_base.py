# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-24  09:35
# NAME:FT-get_base.py
from config import DCC
from data.dataApi import get_indicator
import os
import pandas as pd
import pickle

def get_index():
    index=get_indicator('equity_selected_trading_data').index
    # with open(os.path.join(DCC,'index.pkl'),'wb') as f:
    #     pickle.dump(index,f)
    return index

def parse_ftmt():
    fdmt=get_indicator('equity_fundamental_info')
    fdmt['type_st']=fdmt['type_st'].fillna(0) # nan is not rallowed in fdmt
    index=get_index()
    fdmt=fdmt.reindex(index)
    for col in fdmt.columns:
        fdmt[[col]].to_pickle(os.path.join(DCC,col+'.pkl'))

def parse_others():
    tbnames = ['equity_selected_trading_data',
               'equity_selected_balance_sheet',
               'equity_selected_cashflow_sheet',
               # 'equity_selected_cashflow_sheet_q',
               'equity_selected_income_sheet',
               # 'equity_selected_income_sheet_q',
               ]
    index = get_index()
    for tbname in tbnames:
        df = get_indicator(tbname).reindex(index)
        for col in df.columns:
            df[[col]].to_pickle(os.path.join(DCC, col + '.pkl'))
        print(tbname)

def read_base(indicators):
    if isinstance(indicators,str):
        df=pd.read_pickle(os.path.join(DCC,indicators+'.pkl'))
    else:
        dfs=[pd.read_pickle(os.path.join(DCC,ind+'.pkl')) for ind in indicators]
        df=pd.concat(dfs,axis=1)
    return df

if __name__ == '__main__':
    parse_ftmt()
    parse_others()
