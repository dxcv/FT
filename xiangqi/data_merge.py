# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:24:37 2018

@author: XQZhu
"""
import pandas as pd
# from prerun import signal_analysis
from tools import monitor




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
