# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-30  15:37
# NAME:FT_hp-2 normalize_indicators.py
from empirical.config_ep import DIR_DM_INDICATOR, DIR_DM_NORMALIZED, DIR_DM
import os
import pandas as pd

from tools import multi_process, convert_indicator_to_signal


def _clean_task(name):
    df=pd.read_pickle(os.path.join(DIR_DM_INDICATOR,name,'monthly.pkl'))
    df=df.groupby('month_end').filter(lambda s:len(s.dropna())>300)#trick: at least 300 samples in each month
    if len(df)>0:
        df=convert_indicator_to_signal(df, name)
        df=df.iloc[:,0].unstack().T
        df=df.dropna()#trick: convert_indicator_to_signal may bring in NaNs
        if len(df)>0:
            df.to_pickle(os.path.join(DIR_DM_NORMALIZED,name+'.pkl'))
            print(name)


def clean_indicator():
    names=os.listdir(DIR_DM_INDICATOR)
    print(len(names))
    finished=[f[:-4] for f in os.listdir(os.path.join(DIR_DM_NORMALIZED))]
    names=[n for n in names if n not in finished]
    print(len(names))

    multi_process(_clean_task,names,n=20)

# if __name__ == '__main__':
#     names=open(os.path.join(DIR_DM,'tmp.txt')).read().split('\n')
#     multi_process(_clean_task,names,n=20)

if __name__ == '__main__':
    clean_indicator()

