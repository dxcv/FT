# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-24  11:40
# NAME:FT_hp-build_playing_field.py
import multiprocessing

from config import SINGLE_D_INDICATOR, DIR_TMP
from empirical.config_ep import DIR_KOGAN
from data.dataApi import read_local
import pandas as pd
import numpy as np

import os

from tools import multi_task

G=10

def my_average(df,vname,wname=None):
    '''
    calculate average,allow np.nan in df
    This function intensify the np.average by allowing np.nan

    :param df:DataFrame
    :param vname:col name of the target value
    :param wname:col name of the weights
    :return:scalar
    '''
    if wname is None:
        return df[vname].mean()
    else:
        df=df.dropna(subset=[vname,wname])
        if df.shape[0]>0:
            return np.average(df[vname],weights=df[wname])

fdmt=read_local('equity_fundamental_info')

def get_port_ret(indName):
    print(indName)
    p=os.path.join(DIR_KOGAN,'port_ret','eq',indName+'.pkl')
    if os.path.exists(p):
        return

    try:
        indicator=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, indName + '.pkl'))
        indicator=indicator.stack().swaplevel().sort_index()
        indicator.index.names=['stkcd','trd_dt']
        indicator.name=indName
        # fdmt=read_local('equity_fundamental_info')
        data=pd.concat([fdmt,indicator],axis=1,join='inner')
        data=data.dropna(subset=['type_st','young_1year'])
        data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
        indicator=pd.pivot_table(data,values=indName,index='trd_dt',columns='stkcd')

        indicator=indicator.resample('M').last()
        indicator=indicator.shift(1)
        indicator=indicator.stack().swaplevel().sort_index()

        cap=read_local('fdmt_m')['cap']
        cap=cap.unstack(level='stkcd').shift(1)#trick:
        cap=cap.stack().swaplevel().sort_index()

        ret = read_local('trading_m')['ret_m']
        comb=pd.concat([indicator,cap,ret],axis=1,keys=[indName,'cap','ret'])
        comb.index.names=['stkcd','month_end']

        comb=comb.dropna()
        comb=comb.groupby('month_end').filter(lambda df:df.shape[0]>G*10)
        #TODO: for some indicators, it may throw a error since all its values are 0. Refer to V__dp 2006-02-28
        comb['g']=comb.groupby('month_end',group_keys=False).apply(
            lambda df:pd.qcut(df[indName],G,labels=['g{}'.format(i) for i in range(1,G+1)]))
        port_ret_eq=comb.groupby(['month_end','g'])['ret'].mean().unstack(level=1)
        port_ret_vw=comb.groupby(['month_end','g']).apply(
            lambda df:my_average(df,'ret',wname='cap')).unstack(level=1)

        for port_ret in [port_ret_eq,port_ret_vw]:
            port_ret.columns=port_ret.columns.astype(str)
            port_ret['tb']=port_ret['g{}'.format(G)]-port_ret['g1']

            # port_ret['tb'].cumsum().plot().get_figure().show()
            if port_ret is port_ret_eq:
                port_ret.to_pickle(os.path.join(DIR_KOGAN,'port_ret','eq',indName+'.pkl'))
            else:
                port_ret.to_pickle(os.path.join(DIR_KOGAN,'port_ret','vw',indName+'.pkl'))
    except:
        print(indName,'wrong!!!')

def get_port_ret_part():
    test = pd.read_csv(os.path.join(DIR_KOGAN, 'test_indicators.csv'),
                       encoding='gbk', index_col=0)
    indNames = test['name']
    multiprocessing.Pool(10).map(get_port_ret,indNames)

def get_port_ret_all():
    fns=os.listdir(SINGLE_D_INDICATOR)
    names=[fn[:-4] for fn in fns]
    multi_task(get_port_ret,names,5)


def debug():
    test = pd.read_csv(os.path.join(DIR_KOGAN, 'test_indicators.csv'),
                       encoding='gbk', index_col=0)
    indNames = test['name']

    # for indName in indNames:
    #     print(indName)
    #     get_port_ret(indName)
    indName='C__est_oper_revenue_FT24M_to_close_chg_20'
    get_port_ret(indName)


# fns=os.listdir(SINGLE_D_INDICATOR)
# names=[fn[:-4] for fn in fns]
# get_port_ret(names[-20])


if __name__ == '__main__':
    get_port_ret_all()






