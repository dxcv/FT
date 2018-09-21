# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-27  21:42
# NAME:FT_hp-a.py
import pandas as pd
import os


directory=r'G:\FT_Users\HTZhang\empirical\data_mining\based_on_gta\analyse\conditional\cache'

fns=os.listdir(directory)
fns=[fn for fn in fns if fn.startswith('idio_12M')]

names=[fn[:-4] for fn in fns]
dfs=[]
for fn in fns:
    df=pd.read_pickle(os.path.join(directory,fn))
    dfs.append(df)

comb=pd.concat(dfs,axis=0,keys=names)
comb=comb.loc[(slice(None),'t'),:]
comb['abs']=comb['high-low'].abs()

comb=comb.sort_values('abs')

(comb['abs']>2).sum()
