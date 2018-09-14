# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-16  11:20
# NAME:FT_hp-singleTools.py
import os
from config import LEAST_CROSS_SAMPLE, SINGLE_D_INDICATOR, DIR_TMP
import pandas as pd
from data.dataApi import read_local
from tools import clean


fdmt = read_local('equity_fundamental_info')[['type_st','young_1year','wind_indcd','cap']]
fdmt=fdmt[(~fdmt['type_st']) & (~fdmt['young_1year'])] # 剔除st 和上市不满一年的数据
fdmt=fdmt.dropna(subset=['wind_indcd'])


def convert_indicator_to_signal(df,name):
    '''
    convert daily indicator to daily signal
    Args:
        df:dataFrame, with index as timestamp and columns as stkcd
        name:

    Returns:

    '''
    df = df.stack().to_frame().swaplevel().sort_index().dropna()
    df.index.names = ['stkcd', 'trd_dt']
    df.columns = [name]
    df = pd.concat([fdmt, df], axis=1, join='inner')

    # df = df.dropna(subset=['type_st', 'young_1year'])
    # df = df[(~df['type_st']) & (~ df['young_1year'])]  # 剔除st 和上市不满一年的数据
    # df = df.dropna(subset=['wind_indcd', name])


    df = df.groupby('trd_dt').filter(
        lambda x: x.shape[0] > LEAST_CROSS_SAMPLE)

    # MAD,z-score,neutralized
    df = clean(df, col=name, by='trd_dt')

    signal = pd.pivot_table(df, values=name, index='trd_dt',
                            columns='stkcd').sort_index()
    signal = signal.shift(1)  # trick: we have shift the signal at this place, so 'delay_num' should be 0 in the backtest module!!!!!
    signal=signal.dropna(how='all')
    return signal

