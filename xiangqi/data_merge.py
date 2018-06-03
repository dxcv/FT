# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:24:37 2018

@author: XQZhu
"""
import pandas as pd
# from prerun import signal_analysis
from tools import monitor


@monitor
def factor_merge(df1, df2):
    '''
    df1, df2: DataFrame
        因子购建需要的数据
    去除含有ST标记或上市不满一年的股票
    '''
    keys = ['stkcd', 'trd_dt']
    data = pd.merge(df1, df2.reset_index(), on=keys, how='left')
    #TODO:data=pd.merge(df1,df2,on=keys,right_index=True,how='left')

    # data=data[-int(data.shape[0]/20):]#TODO: use a small sample to test the codes
    data = data.groupby('stkcd').ffill().dropna() #TODO: wrong!! there should be a thresh
    #TODO: warning,just use the last day to determine whether a stock is st or not is not proper
    data = data.groupby('stkcd').resample('M', on='trd_dt').last()
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
