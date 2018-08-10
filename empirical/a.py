# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-31  20:19
# NAME:FT_hp-a.py


import os
import pandas as pd
import numpy as np

path=r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_summary\summary.csv'

df=pd.read_csv(path)

def func(s):
    a,b=s.split('_')[1],s.split('_')[2]
    if a=='iw1' and b=='cw1':
        return True
    else:
        return False
test=df[[func(i) for i in df.iloc[:,0].values]]


# test.to_csv(r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_summary\test.csv')


ss=[]
for name in ['alpha','s1','s2']:
    s=pd.read_csv(os.path.join(r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_summary',name+'.csv'),index_col=0,parse_dates=True).iloc[:,0]
    if name=='alpha':
        s/=100
    s.name=name
    ss.append(s)

df=pd.concat(ss,axis=1)
df=df[:'2017']
corr=df.corr()
print(corr)

