# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-21  17:07
# NAME:FT-dataApi.py
import datetime
import os
import pandas as pd
import pymysql
from config import DRAW, DPKL


def read_raw(tbname):
    return pd.read_csv(os.path.join(DRAW, tbname + '.csv'),index_col=0)

def read_local_pkl(tbname, col=None):
    df=pd.read_pickle(os.path.join(DPKL,tbname+'.pkl'))
    if col:
        if isinstance(col, str):# read only one column
            return df[[col]]
        else: #read multiple columns
            return df[col]
    else: # read all columns
        return df


def download_from_server(tbname,database='filesync'):
    try:
        db = pymysql.connect('192.168.1.140', 'ftresearch', 'FTResearch',
                             database,charset='utf8')
    except:
        db=pymysql.connect('localhost','root','root',database,charset='utf8')
    cur = db.cursor()
    query = 'SELECT * FROM {}'.format(tbname)
    cur.execute(query)
    table = cur.fetchall()
    cur.close()
    table = pd.DataFrame(list(table),columns=[c[0] for c in cur.description])
    table.to_csv(os.path.join(DRAW, '{}.csv'.format(tbname)))

def read_local_sql(tbname,cols=None,database='ft_zht'):
    db = pymysql.connect('localhost', 'root', 'root', database, charset='utf8')
    cur = db.cursor()
    query = 'SELECT * FROM {}'.format(tbname)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    df = pd.DataFrame(list(data),columns=[c[0] for c in cur.description])
    df.columns=[c.lower() for c in df.columns]

    if cols is None:
        return df
    elif isinstance(cols,str):
        return df[[cols]]
    else:
        return df[cols]

def compare_different_database():
    balance_ft=read_local_sql('equity_selected_balance_sheet',database='ftresearch')
    for col in ['trd_dt','ann_dt','report_period']:
        balance_ft[col]=pd.to_datetime(balance_ft[col])
    balance_ft=balance_ft[['stkcd','report_period','ann_dt']]

    # balance_ft=balance_ft.set_index(['stkcd','report_period'])

    balance_file=read_local_sql('asharebalancesheet',database='filesync')
    cols=['wind_code','ann_dt','report_period','actual_ann_dt','statement_type']
    cols_upper=[c.upper() for c in cols]
    balance_file=balance_file[cols_upper]
    balance_file.columns=cols
    balance_file=balance_file.rename(columns={'wind_code': 'stkcd'})
    for col in ['ann_dt','report_period','actual_ann_dt']:
        balance_file[col]=pd.to_datetime(balance_file[col])

    balance_file=balance_file[balance_file['statement_type'].isin(['408001000','408004000',
                                                                   '408005000','408050000'])]


    balance_zht=read_local_sql('asharebalancesheet',database='ft_zht')
    balance_zht=balance_zht[['stkcd','report_period','trd_dt','ann_dt','statement_type']]
    # balance_zht=balance_zht.set_index(['stkcd','report_period'])

    #compare ft with zht
    zht_ft=pd.merge(balance_zht,balance_ft,on=['stkcd','report_period'])
    zht_ft['marker']=zht_ft['ann_dt_x']==zht_ft['ann_dt_y']
    show=zht_ft[~zht_ft['marker']]



    #compare ft with file
    comb=pd.merge(balance_file,balance_ft,on=['stkcd','report_period'],how='outer',
                  indicator=True)

    comb=comb[['stkcd','report_period','ann_dt_x','ann_dt_y','actual_ann_dt','_merge','statement_type']]
    comb=comb.sort_values(['stkcd','report_period'])

    target=comb[comb['ann_dt_x']!=comb['ann_dt_y']]
    target=target.dropna()

    target['statement_type'].value_counts()

    t4000=target[target['statement_type']=='408001000']

    #TODO: trd_dt should be determined by comparing the time point when we get data
    #TODO: and the time point when market closes

    balance_file=balance_file[balance_file['statement_type'].isin(['408001000','408004000',
                                                                   '408005000','408050000'])]
    balance_file.columns=[col.lower() for col in balance_file.columns]
    groups=list(balance_file.groupby(['wind_code','report_period']))
    index,g=groups[0]

    valids=[]
    tmp=None
    for index,g in groups[-100000:-20000]:
        if g.shape[0]>1 and g['inventories'].unique().shape[0]>2:
            # if g['ann_dt']!=g['actual_ann_dt']:
            print(index,g.shape)
            tmp=g
            valids.append(index)




    '''
    use the data in ft to compare with filesync to see which actual_ann_dt it has 
    used.
    '''


# balance=read_local_pkl('asharebalancesheet')
# income=read_local_pkl('ashareincome')
# comb=pd.concat([balance['ann_dt'],income['ann_dt']],axis=1,keys=['balance','income'])
# comb['marker']=(comb['balance']==comb['income'])
# comb=comb.dropna()
# comb['marker'].sum()
#
# dif=comb[~comb['marker']]


