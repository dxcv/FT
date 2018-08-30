# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-17  00:29
# NAME:FT_hp-HaiTong.py
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from data.dataApi import read_local


DIR=r'G:\FT_Users\HTZhang\haitong'



def outlier(s, k=4.5):
    '''
    Parameters
    ==========
    s:Series
        原始因子值
    k = 3 * (1 / stats.norm.isf(0.75))
    '''
    med = np.median(s) #debug: NaN should be removed before apply this function
    mad = np.median(np.abs(s - med))
    uplimit = med + k * mad
    lwlimit = med - k * mad
    y = np.where(s >= uplimit, uplimit, np.where(s <= lwlimit, lwlimit, s))
    # return pd.DataFrame(y, index=s.index)
    return pd.Series(y, index=s.index)

def z_score(x):
    return (x - np.mean(x)) / np.std(x)


indNames=['size','log_size','bp','cumRet','turnover','amihud','ivol']

def clean_indicators():
    for indname in indNames:
        s=pd.read_pickle(os.path.join(DIR,indname+'.pkl')).stack().swaplevel()
        s.name=indname

        fdmt_m = read_local('fdmt_m')
        data = pd.concat([fdmt_m, s], axis=1, join='inner')
        data.index.names=['stkcd','month_end']

        data = data.dropna(subset=['type_st', 'young_1year'])
        data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
        data=data.dropna(subset=[indname])
        data[indname+'_out']=data.groupby('month_end')[indname].apply(outlier)
        data[indname+'_zsc']=data.groupby('month_end')[indname+'_out'].apply(z_score)
        cleaned=pd.pivot_table(data,indname+'_zsc','month_end','stkcd')
        cleaned.to_pickle(os.path.join(DIR,'standardized',indname+'.pkl'))
        print(indname)


