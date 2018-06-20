# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  17:30
# NAME:FT-new_calculate_by_use_excel.py
import pandas as pd
from config import SINGLE_D_INDICATOR
from data.dataApi import get_dataspace
import os


path=r'D:\app\python36\zht\internship\FT\singleFactor\factors\indicators.xlsx'
df=pd.read_excel(path,sheet_name='quality',index_col=0)


def save_indicator(df,name):
    df[['trd_dt',name]].to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))


def parse(name,f,numberator,denominator,arg=None):
    df=get_dataspace([numberator,denominator])
    df[name]=eval(f)(df,numerator,denominator,arg)
    save_indicator(df,name)





for _,s in df.iterrows():
    name=s['name']
    numerator=s['numerator']
    denominator=s['denominator']
    func=s['function']
    arg=s['arg']
    eval(func)()