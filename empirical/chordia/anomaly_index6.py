# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-10  22:24
# NAME:FT_hp-anomaly_index6.py
from config import DIR_TMP
from data.dataApi import get_filtered_ret
from empirical.chordia.conditonal2 import get_comb_indicators
import pandas as pd
import numpy as np
import os

from empirical.config_ep import DIR_BASEDATA
from tools import multi_process


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

def get_anomaly_index(critic):
    indicators=get_comb_indicators(critic)
    indicators1=indicators.dropna(thresh=indicators.shape[1]*0.5)
    indicators1=indicators1.fillna(0)
    anomaly_index=indicators1.sum(axis=1)  #Todo:standardize again before take average
    anomaly_index.name='anomaly_index'
    return anomaly_index

def analyze_univariate_sorting():
    anomaly_index=get_anomaly_index(critic=3)
    ret=get_filtered_ret().swaplevel()

    anomaly_index=anomaly_index.groupby('stkcd').shift(1)
    comb=pd.concat([ret,anomaly_index],axis=1)
    comb=comb.dropna()

    comb.index.names=['month_end','stkcd']

    G=10
    comb=comb.groupby('month_end').filter(lambda df:df.shape[0]>G*10)

    comb['g']=comb.groupby('month_end',group_keys=False).apply(lambda df:pd.qcut(df['anomaly_index'],G,labels=['g{}'.format(i) for i in range(1,G+1)]))
    port_ret_eq=comb.groupby(['month_end','g'])['ret_m'].mean().unstack(level=1)

    s=port_ret_eq['g10']-port_ret_eq['g1']
    (1+s).cumprod().plot().get_figure().show()


def double_sort_dependent(cond_variable):
    conditional=pd.read_pickle(os.path.join(DIR_BASEDATA,'normalized_conditional',cond_variable+'.pkl'))
    anomaly_index=get_anomaly_index(critic=3)

    conditional=conditional.groupby('stkcd').shift(1)
    anomaly_index=anomaly_index.groupby('stkcd').shift(1)
    ret = get_filtered_ret().swaplevel()

    comb=pd.concat([conditional,anomaly_index,ret],axis=1)
    comb=comb.dropna()
    comb=comb.groupby('month_end').filter(lambda df:df.shape[0]>300)#trick: filter out months with too small sample

    #groupby conditional variable
    comb['gc'] = comb.groupby('month_end', group_keys=False).apply(
        lambda df: pd.qcut(df[cond_variable], 5,
                           labels=[f'g{i}' for i in range(1, 6)]))

    # groupby factor
    comb['gf'] = comb.groupby(['month_end', 'gc'], group_keys=False).apply(
        lambda df: pd.qcut(df['anomaly_index'].rank(method='first'), 10,
                           labels=[f'g{i}' for i in range(1, 11)]))

    stk = comb.groupby(['month_end', 'gc', 'gf']).apply(
        lambda df: df['ret_m'].mean()).unstack('gf')
    panel = (stk['g10'] - stk['g1']).unstack()
    panel.columns = panel.columns.astype(str)
    panel['all'] = panel.mean(axis=1)
    panel['high-low'] = panel['g5'] - panel['g1']

    alpha = panel.mean()
    # t=panel.mean()/panel.std()
    t = panel.mean() / panel.sem()  # trick: tvalue = mean / stderr,   stderr = std / sqrt(n-1) ,pd.Series.sem() = pd.Series.std()/pow(len(series),0.5)

    table = pd.concat([alpha, t], axis=1, keys=['alpha', 't']).T
    print(cond_variable)
    return table,cond_variable

def test_all():
    fns = os.listdir(os.path.join(DIR_BASEDATA, 'normalized_conditional'))
    results=multi_process(double_sort_dependent,[fn[:-4] for fn in fns])
    table=pd.concat([r[0] for r in results],axis=0,keys=[r[1] for r in results])
    table.to_csv(os.path.join(DIR_TMP,'table.csv'))

def analyze():
    table=pd.read_csv(os.path.join(DIR_TMP,'table.csv'))
    table.columns=['var','sta']+[col for col in table.columns[2:]]
    table=table[table['sta']=='t'].sort_values('high-low')

def main():
    analyze_univariate_sorting()
    test_all()





# if __name__ == '__main__':
#     test_all()



