# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-30  15:38
# NAME:FT_hp-3 indicator_to_port_ret.py
import os
from data.dataApi import read_local
from empirical.config_ep import DIR_DM_INDICATOR, DIR_DM
from tools import multi_process
import pandas as pd
import numpy as np


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

def _task(args):
    fdmt,cap,indName=args
    # path=os.path.join(DIR_DM_RESULT,indName,'monthly.pkl')
    # if not os.path.exists(path):#fixme:
    #     return

    indicator = pd.read_pickle(
        os.path.join(DIR_DM_INDICATOR, indName,'monthly.pkl'))
    # indicator = indicator.stack().swaplevel().sort_index()
    # indicator.index.names = ['stkcd', 'trd_dt']
    # indicator.name = indName
    data = pd.concat([fdmt, indicator], axis=1, join='inner')
    data = data.dropna(subset=['type_st', 'young_1year'])
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    indicator = pd.pivot_table(data, values=indName, index='trd_dt',
                               columns='stkcd') #review: we can read filtered ret_m directly to simply this function
    indicator = indicator.shift(1)  # trick: use the indicator of time t-1
    indicator = indicator.stack().swaplevel().sort_index()

    ret = read_local('trading_m')['ret_m']
    comb = pd.concat([indicator, cap, ret], axis=1, keys=[indName, 'cap', 'ret'])
    comb.index.names = ['stkcd', 'month_end']

    comb = comb.dropna()
    comb = comb.groupby('month_end').filter(lambda df: df.shape[0] > G * 10)
    # TODO: for some indicators, it may throw a error since all its values are 0. Refer to V__dp 2006-02-28
    if len(comb)>0:
        try:
            comb['g'] = comb.groupby('month_end', group_keys=False).apply(
                lambda df: pd.qcut(df[indName], G,
                                   labels=['g{}'.format(i) for i in range(1, G + 1)],duplicates='raise'))
        except ValueError:
            #trick:qcut with non unique values https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
            comb['g']=comb.groupby('month_end',group_keys=False).apply(
            lambda df:pd.qcut(df[indName].rank(method='first'),G,labels=['g{}'.format(i) for i in range(1,G+1)])
            )
        port_ret_eq = comb.groupby(['month_end', 'g'])['ret'].mean().unstack(level=1)
        port_ret_vw = comb.groupby(['month_end', 'g']).apply(
            lambda df: my_average(df, 'ret', wname='cap')).unstack(level=1)

        for port_ret in [port_ret_eq, port_ret_vw]:
            port_ret.columns = port_ret.columns.astype(str)
            port_ret['tb'] = port_ret['g{}'.format(G)] - port_ret['g1']

            # port_ret['tb'].cumsum().plot().get_figure().show()
            if port_ret is port_ret_eq:
                port_ret.to_pickle(
                    os.path.join(DIR_DM, 'port_ret', 'eq', indName + '.pkl'))
            else:
                port_ret.to_pickle(
                    os.path.join(DIR_DM, 'port_ret', 'vw', indName + '.pkl'))
    print(indName)

def get_port_ret():
    fdmt = read_local('fdmt_m')
    cap = read_local('fdmt_m')['cap']
    cap = cap.unstack(level='stkcd').shift(1)  # trick:use the cap of time t-1
    cap = cap.stack().swaplevel().sort_index()

    names = os.listdir(DIR_DM_INDICATOR)
    print(len(names))
    # names=[fn for fn in names if os.path.exists(os.path.join(DIR_DM_RESULT,fn,'monthly.pkl'))]
    checked = [fn[:-4] for fn in
               os.listdir(os.path.join(DIR_DM, 'port_ret', 'eq'))]
    names = [n for n in names if n not in checked]
    print(len(names))
    args_generator=((fdmt,cap,indName) for indName in names)
    # multi_process(_task, args_generator, 15)
    for args in args_generator:
        _task(args)

if __name__ == '__main__':
    get_port_ret()
