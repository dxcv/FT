# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-01  09:33
# NAME:FT-summarize_single_factors.py

from config import SINGLE_D_RESULT
import os
import pandas as pd
import matplotlib.pyplot as plt

fnames=os.listdir(SINGLE_D_RESULT)

bt_des_dfs=[]
bt_dfs=[]
layer_retn_dfs=[]
layer_des_dfs=[]
for fn in fnames:
    path=os.path.join(SINGLE_D_RESULT,fn)
    btic_des=pd.read_csv(os.path.join(path,'btic_des.csv'),index_col=0)
    btic_m=pd.read_csv(os.path.join(path,'btic_m.csv'),index_col=0)
    layer_retn=pd.read_csv(os.path.join(path,'layer_retn.csv'),index_col=0)
    layer_des=pd.read_csv(os.path.join(path,'layer_des.csv'),index_col=0)

    bt_des_dfs.append(btic_des)
    bt_dfs.append(btic_m)
    layer_retn_dfs.append(layer_retn)
    layer_des_dfs.append(layer_des)

comb_bt_des=pd.concat(bt_des_dfs,axis=1).T
ic_corr=pd.concat([d['ic'] for d in bt_dfs],axis=1,keys=fnames).corr()
tb_des=pd.concat([d['t_b'] for d in layer_des_dfs],axis=1,keys=fnames).T
tb_ret=pd.concat([d['t_b'] for d in layer_retn_dfs],axis=1,keys=fnames)
tb_cret=(tb_ret+1).cumprod()
tb_cret.plot()
plt.show()


