# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-17  18:32
# NAME:FT_hp-0 prepare_data.py
import pandas as pd
import os
import numpy as np

from empirical.config_ep import DIR_DM_GTA
from tools import multi_process

END='2018-08-31'

def get_fdmt_d():
    trading=pd.read_pickle(os.path.join(DIR_DM_GTA,'trading_d.pkl'))
    trading=trading[trading['stkcd'].map(lambda x:x[0] in ['0','3','6'])] #trick: filter stkcds
    trading=trading.set_index(['stkcd','trddt'])

    firm_info=pd.read_pickle(os.path.join(DIR_DM_GTA,'firm_info.pkl'))
    listdt=firm_info.set_index('stkcd')['listdt']
    fdmt=trading.join(listdt, on='stkcd')
    fdmt=fdmt.reset_index()
    fdmt['young_1year']= fdmt['trddt'] <= fdmt['listdt'] + pd.DateOffset(years=1)
    fdmt=fdmt[fdmt['trddt']<=END]
    fdmt=fdmt[fdmt['trdsta'] == 1] #trick：only keep stocks traded normally
    fdmt=fdmt[~fdmt['young_1year']] #trick:
    fdmt=fdmt.set_index(['trddt', 'stkcd']).sort_index()
    col_map={
        'opnprc':'open',
        'hiprc':'high',
        'loprc':'low',
        'clsprc':'close',
        'adjprcwd':'adjclose',
        'dnvaltrd':'amount',
        'dsmvosd':'cap_free',#free float market capitalization
        'dsmvtll':'cap_total',
    }

    fdmt=fdmt.rename(columns=col_map)
    fdmt.to_pickle(os.path.join(DIR_DM_GTA,'fdmt_d.pkl'))

def get_fdmt_m():
    fdmt_d=pd.read_pickle(os.path.join(DIR_DM_GTA, 'fdmt_d.pkl'))
    adjclose=fdmt_d.pivot_table(values='adjclose', index='trddt', columns='stkcd')
    adjclose=adjclose.resample('M').last()

    ret_m=adjclose.pct_change()
    cap1=fdmt_d.pivot_table(values='cap_free',index='trddt',columns='stkcd').resample('M').last()
    cap2=fdmt_d.pivot_table(values='cap_total',index='trddt',columns='stkcd').resample('M').last()
    fdmt_m=pd.concat([ret_m.stack(),cap1.stack(),cap2.stack()],axis=1,keys=['ret_m','cap_free','cap_total']).sort_index()
    fdmt_m.index.names=['month_end','stkcd']
    fdmt_m.to_pickle(os.path.join(DIR_DM_GTA,'fdmt_m.pkl'))

save_df=lambda df,name:df.to_pickle(os.path.join(DIR_DM_GTA,name+'.pkl'))
read_df=lambda name:pd.read_pickle(os.path.join(DIR_DM_GTA,name+'.pkl'))

def yearly2monthly(df):
    '''

    Args:
        df:DataFrame, with single index, panel

    Returns:

    '''
    dr = pd.date_range(start='1990', end='2018-08-31', freq='M')
    df = df.reindex(dr).shift(6).ffill(limit=11)
    df = df.dropna(how='all')
    return df

def get_inv():
    financial=pd.read_pickle(os.path.join(DIR_DM_GTA,'financial.pkl'))
    name='A001000000'
    ta=financial[name]
    ta=ta[ta>0]
    inv=ta.unstack('stkcd').pct_change()
    inv=yearly2monthly(inv)
    save_df(inv,'inv')

def get_log_size():
    fdmt_m=read_df('fdmt_m')
    log_size=np.log(fdmt_m['cap_total'])
    log_size=log_size.unstack('stkcd')
    save_df(log_size,'log_size')

def add_suffix_for_stkcd(s):
    firm_info=read_df('firm_info')
    sid_map = {sid[:-3]: sid for sid in firm_info['stkcd'].values}
    s = s.map(lambda x: sid_map[x])
    return s

def get_bm():
    df=pd.read_csv(os.path.join(DIR_DM_GTA,'other_data','STK_MKT_Dalyr.txt'),sep='\t',encoding='gbk',dtype={'Symbol':str})
    df=df[2:]
    df.columns=[col.lower() for col in df.columns]
    df['symbol']=add_suffix_for_stkcd(df['symbol'])
    df['tradingdate']=pd.to_datetime(df['tradingdate'])

    df=df.set_index(['tradingdate','symbol'])
    df.index.names=['trddt','symbol']

    pb=df['pb'].unstack('symbol').sort_index()
    pb=pb.resample('M').last()
    pb=pb.astype(float)
    bm=np.reciprocal(pb)
    save_df(bm,'bm')

def get_turnover():
    df=pd.read_csv(os.path.join(DIR_DM_GTA,'other_data','STK_MKT_Dalyr.txt'),sep='\t',encoding='gbk',dtype={'Symbol':str})
    df=df[2:]
    df.columns=[col.lower() for col in df.columns]
    df['symbol']=add_suffix_for_stkcd(df['symbol'])
    df['tradingdate']=pd.to_datetime(df['tradingdate'])

    df=df.set_index(['tradingdate','symbol'])
    df.index.names=['trddt','symbol']
    df['turnover']=df['turnover'].astype(float)
    turnover=df['turnover'].unstack('symbol').sort_index()
    for day in [10,20,30,60,120,180,300]:
        tr=turnover.rolling(day,min_periods=int(day/2)).mean()
        tr=tr.resample('M').last()
        save_df(tr,f'turnover_{day}')
        print(day)


def get_mkt_ret():
    df=pd.read_csv(os.path.join(DIR_DM_GTA,'other_data','TRD_Cndalym.txt'),sep='\t',encoding='gbk')
    df=df[2:]
    df.columns = [col.lower() for col in df.columns]
    df['markettype']=df['markettype'].astype(str)

    df=df[df['markettype']=='21'] #综合A股和创业板
    df['trddt']=pd.to_datetime(df['trddt'])
    df=df.set_index('trddt')
    df=df[['cdretwdeq','cdretwdos','cdretwdtl']].astype(float)
    save_df(df,'mkt_ret_d')

    monthly=df.resample('M').last()
    save_df(monthly,'mkt_ret_m')


def idioVol(df):
    df=df.dropna(thresh=int(len(df)*0.6), axis=1)
    df=df.fillna(df.mean())

    # first column is the market return
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:]) #beta
    resid=df.values[:,1:]-X.dot(b)# real value - fitted value
    resid_std=np.std(resid,axis=0)
    return pd.Series(resid_std,index=df.columns[1:],name='idiovol')


def get_comb_for_ivol():
    fdmt_d=read_df('fdmt_d')
    adjclose=fdmt_d['adjclose'].unstack('stkcd')
    ret_d=adjclose.pct_change()
    mkt_ret=read_df('mkt_ret_d')

    comb=pd.concat([mkt_ret['cdretwdos'],ret_d],axis=1) #流通市值加权市场收益
    comb=comb.dropna(subset=['cdretwdos'])
    return comb


def _task_ivol(args):
    ivol_comb,month,window=args
    sub = ivol_comb[:month].last(window)
    print(month)
    return idioVol(sub)


def get_ivol():
    ivol_comb=get_comb_for_ivol()

    fdmt_d=read_df('fdmt_d')
    adjclose=fdmt_d['adjclose'].unstack('stkcd')
    ret_d=adjclose.pct_change()
    mkt_ret=read_df('mkt_ret_d')

    comb=pd.concat([mkt_ret['cdretwdos'],ret_d],axis=1) #流通市值加权市场收益
    comb=comb.dropna(subset=['cdretwdos'])

    month_ends=pd.date_range(start=comb.index[0],end=comb.index[-1],freq='M')

    windows=['3M','6M','12M','24M','36M','60M']

    for window in windows:
        args_generator=((ivol_comb,month,window) for month in month_ends)
        idio=pd.concat(multi_process(_task_ivol,args_generator,5),axis=1,sort=True,keys=month_ends).T
        save_df(idio,f'idio_{window}')




def get_mom():
    fdmt=read_df('fdmt_d')
    adjclose=fdmt['adjclose'].unstack('stkcd')
    mom=adjclose.pct_change(periods=11).shift(1)
    save_df(mom,'mom')

def get_op():
    financial=read_df('financial')
    var1 = 'B001300000'  # 营业利润
    var2 = 'A003100000'  # 归属于母公司所有者权益合计
    financial=financial[financial[var2]>0]
    op=financial[var1]/financial[var2]
    op=op.unstack('stkcd')
    op=yearly2monthly(op)
    save_df(op,'op')

def get_roe():
    '''
    refer to G:\backup\code\assetPricing2\data\din.py\get_roe
    Returns:

    '''
    financial=read_df('financial')
    var1 = 'B002000000'  # 净利润
    var2 = 'A003100000'  # 归属于母公司所有者权益合计
    financial=financial[financial[var2]>0]
    roe=financial[var1]/financial[var2]
    roe=roe.unstack('stkcd')
    roe=yearly2monthly(roe)

    save_df(roe,'roe')


def main():
    get_fdmt_d()
    get_fdmt_m()


# if __name__ == '__main__':
#     main()
