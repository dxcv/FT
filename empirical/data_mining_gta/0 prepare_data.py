# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-17  18:32
# NAME:FT_hp-0 prepare_data.py
import pandas as pd
import os

from empirical.config_ep import DIR_DM_GTA

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

def main():
    get_fdmt_d()
    get_fdmt_m()


if __name__ == '__main__':
    main()
