# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-16  18:39
# NAME:FT_hp-2 normalize_indicators.py
from empirical.config_ep import DIR_DM_GTA, CROSS_LEAST
import os
import pandas as pd
from tools import convert_indicator_to_signal, multi_process



def normalize_task(fn):
    name = fn[:-4]
    path=os.path.join(DIR_DM_GTA,'normalized',fn)
    df=pd.read_pickle(os.path.join(DIR_DM_GTA,'indicators_monthly',fn))
    df=df.stack().to_frame()
    df.columns=[name]
    df.index.names=['month_end','stkcd']
    df=df.groupby('month_end').filter(lambda s:len(s.dropna())>CROSS_LEAST)
    if len(df)>0:
        df=convert_indicator_to_signal(df,name)
        df=df[name].unstack()
        df.to_pickle(path)
        print(name)

def normalize_all():
    fns=os.listdir(os.path.join(DIR_DM_GTA,'indicators_monthly'))
    print('total',len(fns))
    handled=os.listdir(os.path.join(DIR_DM_GTA,'normalized'))
    fns=[fn for fn in fns if fn not in handled]
    print('remainder',len(fns))
    multi_process(normalize_task, fns,20)

if __name__ == '__main__':
    normalize_all()




