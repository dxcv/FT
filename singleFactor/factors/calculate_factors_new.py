# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-29  00:09
# NAME:FT-calculate_factors_new.py

from data.dataApi import read_local
from data.get_base import read_base
import data.database_api.database_api as dbi
from singleFactor.factors.test_single_factor import test_single_factor
import pandas as pd
import matplotlib.pyplot as plt

from tools import handle_duplicates





# 单季度营业利润同比增长率
tbname = 'equity_selected_income_sheet_q'
col = 'oper_profit'

df=dbi.get_stocks_data(tbname,[col,'report_period'])
df['report_period']=pd.to_datetime(df['report_period'])

def func(x):
    result=x.resample('Q',on='report_period').first()
    return result

df1=df.groupby('stkcd').apply(func)

x.shape
result.shape

'''
问题：
1. equity_selected_income_sheet_q 中report_period 的顺序不对，那么直接rolling(4).sum()的时候使用的是错误的数据。
例如，000001.SZ，report_period=2006-03-31的公布日期为2007-04-26 和report_period=2007-03-31 公布的日期一样。这
是因为取得数据是“调整”数据吗？

这种情况，在某个时间点，如果有调整得数据就是用调整后得数据，否则使用调整前得数据。
比如，对于2016年的三季报，我们在2007-04-26前，我们应该使用调整前的数据，在此之后，
应该使用调整后的数据，比如在2007-05的时候，我们应该使用2007-04-26 调整后的数据。
'''
