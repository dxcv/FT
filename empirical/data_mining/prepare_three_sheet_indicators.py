# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-16  11:08
# NAME:FT_hp-prepare_three_sheet_indicators.py
import pandas as pd
from config import DIR_TMP
from data.dataApi import read_from_sql, read_local
import os

from data.filesync.adjust_raw_from_filesync import _adjust_ann_dt, \
    _adjust_dtypes, _delete_unuseful_cols

def get_comb():
    tbnames = ['asharebalancesheet', 'ashareincome', 'asharecashflow']
    dfs=[]
    for tbname in tbnames:
        # df=read_from_sql(tbname)
        df=pd.read_pickle(os.path.join(DIR_TMP,f'{tbname}.pkl'))
        df.columns=[s.lower() for s in df.columns]
        df=_adjust_dtypes(df)
        df=_adjust_ann_dt(df,q=False)
        unuseful=['object_id','s_info_windcode','ann_dt','crncy_code','statement_type','opdate','opmode',
                  'actual_ann_dt','comp_type_code','s_info_compcode','unconfirmed_invest_loss']
        df=df.drop(labels=unuseful,axis=1)
        df=df[df['report_period'].dt.month==12]
        df=df.set_index(['wind_code','report_period'])
        dfs.append(df)
        print(tbname)

    comb=pd.concat(dfs,axis=1)
    comb.to_pickle(os.path.join(DIR_TMP,'comb.pkl'))

comb=pd.read_pickle(os.path.join(DIR_TMP,'comb.pkl'))
comb.index.names=['stkcd','year_end']

comb=comb.unstack('year_end')
stks=[ind for ind in comb.index if ind[0] in ['0','3','6']]
comb=comb.loc[stks]

comb=comb.unstack().unstack(level=0)



indicators=comb.columns.tolist()
valids=[]

indicator=indicators[0]
df=comb[indicator].unstack()

test=comb.groupby('year_end').apply(lambda df:df.notnull().sum())
test.to_csv(os.path.join(DIR_TMP,'count.csv'))


# if __name__ == '__main__':
#     get_comb()


# df.columns=[s.lower() for s in df.columns]
# df = df.apply(pd.to_numeric, errors='ignore')
