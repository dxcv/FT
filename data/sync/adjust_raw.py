# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-02  16:31
# NAME:FT-adjust_raw.py

import os
import pandas as pd
import numpy as np

from config import DRAW, DCSV, DPKL
from data.dataApi import read_raw, read_local_pkl, read_local_sql
from tools import number2dateStr

from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR


def save_df(df,name):
    '''
    save df to pkl and mysql

    Args:
        df:pd.DataFrame
        name:str,table name

    Returns:

    '''
    # df.to_csv(os.path.join(DCSV, name + '.csv'))
    df.to_pickle(os.path.join(DPKL, name + '.pkl'))
    '''
    https://techjourney.net/mysql-error-1170-42000-blobtext-column-used-in-key-specification-without-a-key-length/
    https://stackoverflow.com/questions/45285184/python3-pandas-and-mysql-index-issue
    '''
    con = create_engine(
        'mysql+pymysql://root:root@localhost/ft_zht?charset=utf8')
    df.to_sql(name=name,con=con,if_exists='replace',dtype={'stkcd':VARCHAR(9)})
    #TODO: add comments for columns

def adjust_filesync(tbname):
    dateFields = ['report_period', 'trd_dt', 'ann_dt', 'holder_enddate',
                  'listdate','actual_ann_dt']
    # actual_ann_dt 指的是“更正公告的日期”
    df = pd.read_csv(os.path.join(DRAW, tbname + '.csv'), index_col=0)

    #adjust column format
    df.columns=[str.lower(c) for c in df.columns]

    #adjust date format
    for datef in dateFields:
        if datef in df.columns:
            df[datef]=df[datef].map(lambda x:str(x)[:8])
            df[datef] = pd.to_datetime(df[datef])

    #map ann_date to trading date
    calendar=read_raw('asharecalendar')
    trd_dt=pd.to_datetime(calendar['TRADE_DAYS'].map(str)).drop_duplicates().sort_values()
    df['trd_dt']=df['ann_dt'].map(lambda x:trd_dt.values[trd_dt.searchsorted(x)[0]])

    # reindex with qrange
    def _reindex_with_qrange(x):
        x=x.set_index('report_period')
        # qrange=pd.date_range(x.index.min(), x.index.max(), freq='Q')
        # result2=x.reindex(pd.Index(qrange,name='report_period'))
        x=x.resample('Q').asfreq()
        return x

    #handle duplicates,keep the last one
    #keep the first one,since in real life,we can only trade based on the first
    #one.
    df=df.sort_values(['wind_code','report_period','trd_dt'])
    df=df[~df.duplicated(['wind_code','report_period'],keep='first')]
    df=df.groupby('wind_code').apply(_reindex_with_qrange)

    df=df.drop(['object_id','s_info_windcode','wind_code'],axis=1)
    df=df.reset_index().rename(columns={'wind_code':'stkcd'})

    df=df.set_index(['stkcd','report_period'])
    df=df.sort_index()

    #reorder the columns
    newcols=['trd_dt']+[c for c in df.columns if c!= 'trd_dt']
    df=df[newcols]

    df.to_csv(os.path.join(DCSV,tbname+'.csv'))
    df.to_pickle(os.path.join(DPKL,tbname+'.pkl'))

    #TODO:notice that do not apply sort and dropna on df
    #TODO:notice that there are some NaT in index

def convert_asharefinancialindicator():
    tbname='asharefinancialindicator'
    adjust_filesync(tbname)

#-------------------------------parse financial report-------------------------
def _adjust_dtypes(df):
    #TODO: how about using mysql to determine the dtypes？
    df=df.apply(pd.to_numeric,errors='ignore')

    df['opdate']=pd.to_datetime(df['opdate'],unit='ns')
    dateFields = ['report_period', 'trd_dt', 'ann_dt', 'holder_enddate',
                  'listdate','actual_ann_dt']

    for dcol in dateFields:
        if dcol in df.columns:
            df[dcol]=pd.to_datetime(df[dcol].map(number2dateStr ))
    return df


def _adjust_ann_dt(df, q=False):
    #使用合并报表
    #[1000,4000,5000] for accumulative value,[2000,3000] for single quarter value
    #累积数据
    if q:
        df = df[df['statement_type'].isin([408002000, 408003000])]
    else:
        df = df[df['statement_type'].isin([408001000, 408004000, 408005000, 4080050000])]
    df = df.sort_values(['wind_code', 'report_period', 'actual_ann_dt'],
                        ascending=True)
    #TODO: 直接去最小日期也有问题，有的1000 和5000 由相同的最小日期，还要确定具体取哪个statement_type
    df = df[~df.duplicated(subset=['wind_code', 'report_period'],
                           keep='first')]  # 取最早的那条数据，并且使用实际公告日，而不是ann_dt
    '''
    在回测中，为了避免未来函数同时兼顾对更正信息的使用，我们应该在每个时点使用当前
    时点的最新信息。
    这里为了方便，我们只保留一条数据，我们选择保留最早的那条数据（以牺牲更正信息为
    代价保证不引入未来函数），基于以下几点原因：
        1. 使用最早的那条数据肯定是不会引入未来函数
        2. 虽然在回测中我们使用的是最早那条数据而没有用最新更正的数据（这样的样本其
        实也不多）可能在回测中有误差，但是这样的样本都是出现在距离当前时点较远的历史
        数据中，因为那些更正一般都是在公布下一个年报的时候一起公布的，也就是说我们最近
        的数据不可能引入未来函数，虽然我们那些更久远一点的历史数据可能没有用到更正信息
        但是我们保证了最新历史数据的准确性。倘若我们使用的是含有未来函数的数据（比如
        我们用了最新的数据），那么我们虽然保证了使用更正信息，但是我们最近一年左右的
        历史数据却被未来函数污染了。

    '''
    return df

def _add_trd_dt(df):
    calendar=read_raw('asharecalendar')
    trd_dt=pd.to_datetime(calendar['TRADE_DAYS'].map(str)).drop_duplicates().sort_values()
    df['trd_dt']=df['actual_ann_dt'].map(lambda x:trd_dt.values[trd_dt.searchsorted(x)[0]])
    return df

def _fill_report_period(df):
    # reindex with qrange
    def _reindex_with_qrange(x):
        x = x.set_index('report_period')
        # qrange=pd.date_range(x.index.min(), x.index.max(), freq='Q')
        # result2=x.reindex(pd.Index(qrange,name='report_period'))
        x = x.resample('Q').asfreq()
        return x
    df=df.groupby('wind_code').apply(_reindex_with_qrange)
    return df

def _delete_unuseful_cols(df):
    unuseful_cols=['object_id','s_info_windcode','wind_code','ann_dt']
    df=df.drop(labels=unuseful_cols,axis=1)
    return df

def _adjust_df(df):
    #reorder the columns
    df=df.rename(columns={'actual_ann_dt':'ann_dt'}) #adjust actual_ann_dt 与ftresearch的命名方式一致
    hd=['trd_dt','ann_dt']
    new_order=hd+[c for c in df.columns if c not in hd]
    df=df[new_order]
    df.index.names=['stkcd','report_period'] # replace 'wind_code' with 'stkcd',与ftresearch 的命名保持一致
    return df

def _parse(name, q=False):
    df=read_local_sql(name,database='filesync')
    #adjust data type
    # refer to this link:https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas
    df=df.apply(pd.to_numeric,errors='ignore')

    df = _adjust_dtypes(df)
    df=_adjust_ann_dt(df, q)
    df = _fill_report_period(df)
    df = _add_trd_dt(df)
    df = _delete_unuseful_cols(df)
    df = _adjust_df(df)
    if q:
        name=name+'_q'
    save_df(df,name)

def parse_three_reports():
    names=['asharebalancesheet','ashareincome','asharecashflow']
    for name in names:
        _parse(name, q=False)
    for name in names[1:]:
        _parse(name, q=True)

# parse_three_reports()


def calculate_q_sheet():
    #calculate myself q sheet

    ac=read_local_pkl('ashareincome')

    df=ac.reset_index()

    def _adjust(x):
        cols=['stkcd','report_period','trd_dt','ann_dt','statement_type','crncy_code',
              'comp_type_code','s_info_compcode','opdate','opmode']
        indicators=[c for c in x.columns if c not in cols]
        if x.shape[0]==4:
            x[indicators]=x[indicators].apply(lambda s:s-s.shift(1).fillna(0))
            #TODO: wrong ,有缺失数据的时候会出错
            return x

        #TODO: float type rather than str
    single_q=df.groupby(['stkcd',df['report_period'].dt.year]).apply(_adjust)



#TODO: compare my data with ftresearch
#TODO:compare ftresearch with filesync to see which one it has choosen
#TODO: calculate q sheets


#TODO: only keep stock codes  rather than wind_code

'''
ftresearch 的单位进行了调整，比如tot_cur_liab的单位调整为了万元，四舍五入。
'''


