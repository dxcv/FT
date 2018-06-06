# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-01  09:33
# NAME:FT-summarize_single_factors.py

from config import SINGLE_D_RESULT
import os
import pandas as pd
import matplotlib.pyplot as plt

D_RESULT=r'D:\zht\database\quantDb\internship\FT\singleFactor\summary'

def summarize_result():
    fnames=os.listdir(SINGLE_D_RESULT)
    bt_des_dfs=[]
    bt_dfs=[]
    layer_retn_dfs=[]
    layer_des_dfs=[]
    for fn in fnames:
        path=os.path.join(SINGLE_D_RESULT,fn)
        btic_des=pd.read_csv(os.path.join(path,'btic_des.csv'),index_col=0).rename(columns={'target':fn})
        btic_m=pd.read_csv(os.path.join(path,'btic_m.csv'),index_col=0)
        layer_retn=pd.read_csv(os.path.join(path,'layer_retn.csv'),index_col=0)
        layer_des=pd.read_csv(os.path.join(path,'layer_des.csv'),index_col=0)

        bt_des_dfs.append(btic_des)
        bt_dfs.append(btic_m)
        layer_retn_dfs.append(layer_retn)
        layer_des_dfs.append(layer_des)

    bt_des=pd.concat(bt_des_dfs,axis=1).T
    ic_corr=pd.concat([d['ic'] for d in bt_dfs],axis=1,keys=fnames).corr()
    tb_des=pd.concat([d['t_b'] for d in layer_des_dfs],axis=1,keys=fnames).T

    tb_ret=pd.concat([d['t_b'] for d in layer_retn_dfs],axis=1,keys=fnames)
    tb_ret['zz500']=layer_retn_dfs[0]['retn_1m_zz500']

    fig1=plt.figure()
    tb_cret1=(tb_ret+1).cumprod()
    tb_cret1.plot(figsize=(20,6))
    plt.savefig(os.path.join(D_RESULT,'cumprod.pdf'))

    fig2=plt.figure()
    tb_cret2=tb_ret.cumsum()
    tb_cret2.plot(figsize=(20,6))
    plt.savefig(os.path.join(D_RESULT,'cumsum.pdf'))

    bt_des.to_csv(os.path.join(D_RESULT,'bt_des.csv'))
    ic_corr.to_csv(os.path.join(D_RESULT,'ic_corr.csv'))
    tb_des.to_csv(os.path.join(D_RESULT,'tb_des.csv'))
    tb_ret.to_csv(os.path.join(D_RESULT,'tb_ret.csv'))

def get_prominent_factors():
    tb_des=pd.read_csv(os.path.join(D_RESULT,'tb_des.csv'),index_col=0,parse_dates=True)
    tb_ret=pd.read_csv(os.path.join(D_RESULT,'tb_ret.csv'),index_col=0,parse_dates=True)


    non_ttm=[ind for ind in tb_des.index if not ind.endswith('_ttm')]
    nl=tb_des.loc[non_ttm]['sharpe'].nlargest(10).index
    ns=tb_des.loc[non_ttm]['sharpe'].nsmallest(10).index

    df_nl=tb_ret[nl.tolist()].dropna()
    df_ns=tb_ret[ns.tolist()].dropna()

    # df_nl=tb_ret[nl.tolist()+['zz500']].dropna()
    # df_ns=tb_ret[ns.tolist()+['zz500']].dropna()


    figsize=(20,6)

    fig1=plt.figure()
    cp=(df_nl+1).cumprod()
    cp.plot(figsize=figsize)
    plt.savefig(os.path.join(D_RESULT,'positive_cumprod.png'))


    plt.figure()
    cp=(df_ns+1).cumprod()
    cp.plot(figsize=figsize)
    plt.savefig(os.path.join(D_RESULT,'negative_cumprod.png'))















