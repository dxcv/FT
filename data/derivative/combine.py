# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-15  13:42
# NAME:FT-combine.py
import os
import pickle

from config import DCC
from data.dataApi import read_local
import pandas as pd


def _get_fields_map():
    tbs_ftresearch=[
    'equity_selected_balance_sheet',
    # 'equity_selected_cashflow_sheet',
    'equity_selected_cashflow_sheet_q',
    # 'equity_selected_income_sheet',
    'equity_selected_income_sheet_q',
    'equity_cash_dividend',
    ]

    tbs_filesync=['asharefinancialindicator']

    shared_cols=['stkcd','trd_dt','ann_dt','report_period']
    fields_map={}
    for tbname in tbs_ftresearch+tbs_filesync:
        try:
            df=read_local(tbname,src='ftresearch')
        except:
            df=read_local(tbname,src='filesync')
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
        df=read_local(dfnames[0])
    else:
        df=pd.concat([read_local(dn) for dn in dfnames], axis=1)

    if isinstance(df['trd_dt'],pd.DataFrame):
        '''如果df里边有多个名为trd_dt的列，取日期最大的那个'''
        trd_dt_df=df['trd_dt']
        df=df.drop('trd_dt',axis=1)
        df['trd_dt']=trd_dt_df.max(axis=1)
    return df[['trd_dt']+fields]


