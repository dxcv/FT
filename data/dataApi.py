# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-21  17:07
# NAME:FT-dataApi.py
import datetime
import os
from functools import reduce

import pandas as pd
import pickle
import pymysql
from config import DRAW, DPKL, D_FT_ADJ, D_FILESYNC_ADJ, DCC, D_DRV


def read_raw(tbname):
    return pd.read_csv(os.path.join(DRAW, tbname + '.csv'),index_col=0)

def read_local(tbname,col=None):
    #DEBUG: 不能这么弄，这样会导致数据很混乱

    #TODO: rewrite this function
    df=None
    for d in [D_FT_ADJ,D_FILESYNC_ADJ,D_DRV]:
        path=os.path.join(d,tbname+'.pkl')
        if os.path.exists(path):
            df=pd.read_pickle(path)
            break

    if col:
        if isinstance(col, str):# read only one column
            return df[[col]]
        else: #read multiple columns
            return df[col]
    else: # read all columns
        return df


def read_local_pkl(tbname, col=None):
    df=pd.read_pickle(os.path.join(DPKL,tbname+'.pkl'))
    if col:
        if isinstance(col, str):# read only one column
            return df[[col]]
        else: #read multiple columns
            return df[col]
    else: # read all columns
        return df


def read_from_sql(tbname, database='filesync'):
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
    return table

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


def _get_fields_map():
    tbs_ftresearch=[
    'equity_selected_balance_sheet',
    # 'equity_selected_cashflow_sheet',
    'equity_selected_cashflow_sheet_q',
    # 'equity_selected_income_sheet',
    'equity_selected_income_sheet_q',
    'equity_cash_dividend',
    'shr_and_cap'
    ]
    
    tbs_derivatives=['ebit','grossIncome','indice_m','netAsset','NetNonOI',
                     'payable','periodCost','receivable']
    # tbs_derivatives=[f[:-4] for f in os.listdir(D_DRV)]
    tbs_filesync=['asharefinancialindicator']

    shared_cols=['stkcd','trd_dt','ann_dt','report_period']
    fields_map={}
    for tbname in tbs_ftresearch+tbs_derivatives+tbs_filesync:
        df=read_local(tbname)
        indicators=[col for col in df.columns if col not in shared_cols]
        for ind in indicators:
            if ind not in fields_map.keys():
                fields_map[ind]=tbname
            else:
                raise ValueError('Different tables share the indicator -> "{}"'.format(ind))
    #TODO: cache for fields_map

    return fields_map

def read_fields_map(refresh=True):
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


def mix_dfs(dfs):
    '''combine dfs,different df may have different frequency
    For example:
        when we combine 'shr_and_cap' (monthly) with 'equity_selected_balance_sheet' (quartly),
        the frequency of the returned df will be quarterly. And the index names will
        become ['stkcd','report_period']
    '''
    df=pd.concat(dfs,axis=1,join='inner')
    trd_dt_df=df['trd_dt']
    dt=trd_dt_df.max(axis=1)
    del df['trd_dt']
    df['trd_dt']=dt

    names=[]
    for a in dfs:
        for name in a.index.names:
            if name not in names:
                names.append(name)

    if 'report_period' in names:
        df.index.names=['stkcd','report_period']

    #TODO: other situations?
    return df

def get_dataspace(fields):
    #TODO: how to combine shr_and_cap,dequity_selected_trading_data
    # their index are ['stkcd','trd_dt']
    fields_map=_get_fields_map()
    if isinstance(fields,str): #only one field
        fields=[fields]

    dfnames=list(set([fields_map[f] for f in fields]))
    if len(dfnames)==1:
        df=read_local(dfnames[0])
    else:
        df=mix_dfs([read_local(dn) for dn in dfnames])
    return df[['trd_dt']+fields]

def check_dfs():
    for d in [D_FT_ADJ,D_FILESYNC_ADJ,D_DRV]:
        fns=os.listdir(d)
        for fn in fns:
            df=pd.read_pickle(os.path.join(d,fn))
            print(fn,df.index.names)


#-------------------------------20180622---------------------------------------





#get_mould_index

# fp1=r'D:\zht\database\quantDb\internship\FT\database\filesync_based\adjusted\trading_m.pkl'
# fp2=r'D:\zht\database\quantDb\internship\FT\database\filesync_based\adjusted\shr_and_cap.pkl'
# fp3=r'D:\zht\database\quantDb\internship\FT\database\ftresearch_based\adjusted\pkl\equity_cash_dividend.pkl'
#
# df1=pd.read_pickle(fp1)
# df2=pd.read_pickle(fp2)
# df3=pd.read_pickle(fp3)
#
# dfs=[df.reset_index() for df in [df1,df2,df3]]
#
# df_merged=reduce(lambda left,right:pd.merge(left,right,on=['stkcd','trd_dt'],
#                                             how='outer'),dfs)
#
#

