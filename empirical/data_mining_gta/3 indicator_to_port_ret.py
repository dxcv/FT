# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-16  19:16
# NAME:FT_hp-3 indicator_to_port_ret.py

import os
from empirical.config_ep import DIR_DM_GTA, CROSS_LEAST, PERIOD_THRESH
from tools import multi_process
import pandas as pd
import numpy as np

DIR_NORMALIZED=os.path.join(DIR_DM_GTA,'normalized')


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
    cap,ret,indName=args
    indicator=pd.read_pickle(os.path.join(DIR_NORMALIZED,indName+'.pkl')).shift(1).stack()#trick: shift(1), use indicator in time t-1 to construct portfolio in time t
    cap=cap.groupby('stkcd').shift(1) #trick: use the cap of time t-1 as weight
    '''
    cap has been filtered, as long as we concat the dropna(), it is the same to filter on any indicator.
    '''
    comb = pd.concat([indicator, cap, ret], axis=1, keys=[indName, 'cap', 'ret'])
    comb.index.names = ['month_end','stkcd']

    comb = comb.dropna()
    comb = comb.groupby('month_end').filter(lambda df: df.shape[0] > CROSS_LEAST)
    # TODO: for some indicators, it may throw a error since all its values are 0. Refer to V__dp 2006-02-28

    if len(set(comb.index.get_level_values('month_end')))>=PERIOD_THRESH:
    # if len(comb)>0:
        try:
            comb['g'] = comb.groupby('month_end', group_keys=False).apply(
                lambda df: pd.qcut(df[indName], G,
                                   labels=['g{}'.format(i) for i in range(1, G + 1)],duplicates='raise'))
        except ValueError:
            # def test(df):
            #     result=pd.qcut(df[indName].rank(method='first'),G,labels=['g{}'.format(i) for i in range(1,G+1)])
            #     return result
            #
            # comb['g']=comb.groupby('month_end',group_keys=False).apply(test)


            # trick:qcut with non unique values https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
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
                    os.path.join(DIR_DM_GTA, 'port_ret', 'eq', indName + '.pkl'))
            else:
                port_ret.to_pickle(
                    os.path.join(DIR_DM_GTA, 'port_ret', 'vw', indName + '.pkl'))
    print(indName)
#
# def task(args):#fixme:
#     try:
#         _task(args)
#     except:
#         print('wrong--------------->',args[2])
#         pass

def get_port_ret():
    fdmt_m=pd.read_pickle(os.path.join(DIR_DM_GTA,'fdmt_m.pkl'))
    cap=fdmt_m['cap_total']# trick: use total market capitalization as value weight
    ret=fdmt_m['ret_m']
    names = [fn[:-4] for fn in os.listdir(DIR_NORMALIZED)]
    print(len(names))
    # names=[fn for fn in names if os.path.exists(os.path.join(DIR_DM_RESULT,fn,'monthly.pkl'))]
    checked = [fn[:-4] for fn in
               os.listdir(os.path.join(DIR_DM_GTA, 'port_ret', 'eq'))]
    names = [n for n in names if n not in checked]
    print(len(names))

    args_generator=((cap,ret,indName) for indName in names)
    multi_process(_task, args_generator, 15,size_in_each_group=500)

    # for args in args_generator:
    #     _task(args)

def debug():
    fdmt_m = pd.read_pickle(os.path.join(DIR_DM_GTA, 'fdmt_m.pkl'))
    cap = fdmt_m[
        'cap_total']  # trick: use total market capitalization as value weight
    ret = fdmt_m['ret_m']
    indName='ratio_growth_dif-A001219000-A002200000'
    _task((cap,ret,indName))


if __name__ == '__main__':
    get_port_ret()
    # debug()
