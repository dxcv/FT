# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-05  13:50
# NAME:FT-avg.py
import os

import pandas as pd
from config import SINGLE_D_CHECK, SINGLE_D_INDICATOR, DIR_CLEANED
from data.dataApi import read_local
from tools import filter_st_and_young, clean

def get_cleaned_factors():
    path = r'D:\app\python36\zht\internship\FT\singleFactor\indicators.xlsx'
    df = pd.read_excel(path, sheet_name='valid', encoding='gbk')
    for indicator in df['name']:
        df=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR,indicator+'.pkl'))
        fdmt = read_local('fdmt_m')[
            ['cap', 'type_st', 'wind_indcd', 'young_1year']]
        # ret_1m = read_local('trading_m')['ret_1m']
        # zz500_ret_1m=read_local('indice_m')['zz500_ret_1m']
        monthly=filter_st_and_young(df,fdmt)
        monthly = monthly.dropna()
        monthly = monthly.groupby('month_end').filter(
            lambda x: x.shape[0] > 300)  # trick: filter,因为后边要进行行业中性化，太少的样本会出问题

        monthly = clean(monthly,indicator)  # outlier,z_score,nutrualize
        # comb = pd.concat([ret_1m, monthly[indicator]], axis=1, join='inner').dropna()
        monthly[[indicator]].to_pickle(os.path.join(DIR_CLEANED,indicator+'.pkl'))
        print(indicator)

get_cleaned_factors()








