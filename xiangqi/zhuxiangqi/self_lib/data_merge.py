# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:24:37 2018

@author: XQZhu
"""
import pandas as pd
# from prerun import signal_analysis

def factor_merge(df1, df2, keys=['stkcd', 'trd_dt']):
    '''
    df1, df2: DataFrame
        因子购建需要的数据
    去除含有ST标记或上市不满一年的股票
        
    '''
    data = pd.merge(df1, df2, on=keys, how='left')
    data = data.groupby(keys[0]).ffill().dropna()
    data = data.groupby(keys[0]).resample('M', on=keys[1]).last()
#    data = data.set_index(['stkcd', 'trd_dt'], drop=False)
    data = data[(data.type_st == 0) & (data.year_1 == 0)]
    return data

# 利用公司测试框架需要的数据格式
def test_merge(data, f, cp, cp_name='close_price_post', index=['trd_dt', 'stkcd']):
    price_input = pd.DataFrame(cp.stack(), columns=[cp_name])
    data = data.set_index(index)
    data.index.names = price_input.index.names
    signal_input = data[[f]]
    test_data = price_input.join(signal_input, how='left')
    test_data = test_data.groupby(level=1).ffill().dropna()
    return signal_input , test_data

# def test_result(test_data, f, price):
#     return signal_analysis(test_data[f].unstack(), test_data[price].unstack())
