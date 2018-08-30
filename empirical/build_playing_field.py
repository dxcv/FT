# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-24  11:40
# NAME:FT_hp-build_playing_field.py
import objgraph



from config import SINGLE_D_INDICATOR, DIR_TMP, DIR_DM_RESULT
from empirical.config_ep import DIR_KOGAN
from data.dataApi import read_local
import pandas as pd
import numpy as np

import os

from tools import multi_process,outlier,z_score

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


def get_port_ret(indName):
    fdmt=read_local('equity_fundamental_info')#fixme: send in rather than read again
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
        indicator=indicator.shift(1) #trick: use the indicator of time t-1
        indicator=indicator.stack().swaplevel().sort_index()

        cap=read_local('fdmt_m')['cap']
        cap=cap.unstack(level='stkcd').shift(1)#trick:use the cap of time t-1
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

def get_port_ret1():
    test = pd.read_csv(os.path.join(DIR_KOGAN, 'test_indicators.csv'),
                       encoding='gbk', index_col=0)
    indNames = test['name']
    # multiprocessing.Pool(10).map(get_port_ret,indNames)

def get_port_ret2():
    fns=os.listdir(SINGLE_D_INDICATOR)
    names=[fn[:-4] for fn in fns]
    multi_process(get_port_ret, names, 5)


def _get_port_ret3(args):
    fdmt,cap,indName=args
    # path=os.path.join(DIR_DM_RESULT,indName,'monthly.pkl')
    # if not os.path.exists(path):#fixme:
    #     return

    indicator = pd.read_pickle(
        os.path.join(DIR_DM_RESULT, indName,'monthly.pkl'))
    # indicator = indicator.stack().swaplevel().sort_index()
    # indicator.index.names = ['stkcd', 'trd_dt']
    # indicator.name = indName
    data = pd.concat([fdmt, indicator], axis=1, join='inner')
    data = data.dropna(subset=['type_st', 'young_1year'])
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    indicator = pd.pivot_table(data, values=indName, index='trd_dt',
                               columns='stkcd')
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
                    os.path.join(DIR_KOGAN, 'port_ret', 'eq', indName + '.pkl'))
            else:
                port_ret.to_pickle(
                    os.path.join(DIR_KOGAN, 'port_ret', 'vw', indName + '.pkl'))
    print(indName)


def get_port_ret3():
    fdmt = read_local('fdmt_m')
    cap = read_local('fdmt_m')['cap']
    cap = cap.unstack(level='stkcd').shift(1)  # trick:use the cap of time t-1
    cap = cap.stack().swaplevel().sort_index()

    names = os.listdir(DIR_DM_RESULT)
    print(len(names))
    # names=[fn for fn in names if os.path.exists(os.path.join(DIR_DM_RESULT,fn,'monthly.pkl'))]
    checked = [fn[:-4] for fn in
               os.listdir(os.path.join(DIR_KOGAN, 'port_ret', 'eq'))]
    names = [n for n in names if n not in checked]
    print(len(names))
    args_generator=((fdmt,cap,indName) for indName in names)
    multi_process(_get_port_ret3, args_generator, 15)
    # [_get_port_ret3(args) for args in args_generator]


from memory_profiler import profile
#
fp=open(os.path.join(DIR_TMP,'memory_debug.log'),'a')


@profile(precision=4,stream=fp)
def convert_indicator_to_signal(df, name):
    df[name]=df[name].groupby('month_end').apply(outlier)
    df[name]=df[name].groupby('month_end').apply(z_score)
    return df

def _clean_task(name):
    df=pd.read_pickle(os.path.join(DIR_DM_RESULT,name,'monthly.pkl'))
    df=df.groupby('month_end').filter(lambda s:len(s)>300)#trick: at least 300 samples in each month
    df=convert_indicator_to_signal(df, name)
    df=df.iloc[:,0].unstack().T
    df.to_pickle(os.path.join(DIR_KOGAN,'signal',name+'.pkl'))
    print(name)

def clean_indicator():
    names=os.listdir(DIR_DM_RESULT)
    print(len(names))
    finished=[f[:-4] for f in os.listdir(os.path.join(DIR_KOGAN,'signal'))]
    names=[n for n in names if n not in finished]
    print(len(names))

    multi_process(_clean_task,names,n=20)
    # [_clean_task(name) for name in names]

def tmp_func(name):
    df = pd.read_pickle(os.path.join(DIR_DM_RESULT, name, 'monthly.pkl'))
    df = df.groupby('month_end').filter(
        lambda s: len(s) > 300)  # trick: at least 300 samples in each month
    df[name] = df[name].groupby('month_end').apply(outlier)
    df[name] = df[name].groupby('month_end').apply(z_score)
    df = df.iloc[:, 0].unstack().T
    df.to_csv(os.path.join(DIR_TMP, name + '.csv'))
    print(df.info())
    del df
    # df=pd.DataFrame()
    # gc.collect()

@profile(precision=4,stream=fp)
def debug2():
    names=os.listdir(DIR_DM_RESULT)
    print(len(names))
    finished=[f[:-4] for f in os.listdir(os.path.join(DIR_KOGAN,'signal'))]
    names=[n for n in names if n not in finished]
    print(len(names))
    name=names[0]
    tmp_func(name)
    # df=pd.DataFrame()
    # df.to_pickle(os.path.join(DIR_KOGAN, 'signal', name + '.pkl'))


@profile(precision=4,stream=fp)
def debug():
    names = os.listdir(DIR_DM_RESULT)
    print(len(names))
    finished = [f[:-4] for f in os.listdir(os.path.join(DIR_KOGAN, 'signal'))]
    names = [n for n in names if n not in finished]
    print(len(names))

    _clean_task(names[0])
    _clean_task(names[1])

def debug1():
    names = os.listdir(DIR_DM_RESULT)
    print(len(names))
    finished = [f[:-4] for f in os.listdir(os.path.join(DIR_KOGAN, 'signal'))]
    names = [n for n in names if n not in finished]
    print(len(names))
    name=names[0]
    objgraph.show_growth()
    _clean_task(name)
    objgraph.show_growth()



def main():
    get_port_ret3()
    clean_indicator()


if __name__ == '__main__':
    # get_port_ret3()
    clean_indicator()







