# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-16  11:20
# NAME:FT_hp-singleTools.py



from config import LEAST_CROSS_SAMPLE
import pandas as pd
from data.dataApi import read_local
from tools import clean


fdmt = read_local('equity_fundamental_info')


def convert_indicator_to_signal(df,name):
    df = df.stack().to_frame().swaplevel().sort_index()
    df.index.names = ['stkcd', 'trd_dt']
    df.columns = [name]
    data = pd.concat([fdmt, df], axis=1, join='inner')

    data = data.dropna(subset=['type_st', 'young_1year'])
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    data = data.dropna(subset=['wind_indcd', name])
    data = data.groupby('trd_dt').filter(
        lambda x: x.shape[0] > LEAST_CROSS_SAMPLE)
    cleaned_data = clean(data, col=name, by='trd_dt')
    signal = pd.pivot_table(cleaned_data, values=name, index='trd_dt',
                            columns='stkcd').sort_index()
    signal = signal.shift(1)  # trick:
    signal=signal.dropna(how='all')
    return signal



