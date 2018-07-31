# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-25  08:54
# NAME:FT_hp-build_models.py
import multiprocessing
import os
import itertools
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from config import  DIR_KOGAN
from tools import multi_task


def _generate_models(names):
    directory=os.path.join(DIR_KOGAN,'port_ret','eq')
    factors=[]
    for name in names:
        factor=pd.read_pickle(os.path.join(directory,name+'.pkl'))['tb']
        factor.name=name
        factors.append(factor)

    rpM=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata','rpM.pkl'))
    comb=pd.concat(factors+[rpM],axis=1).dropna()
    comb.to_pickle(os.path.join(DIR_KOGAN,'models',str(len(names)+1),'___'.join(names)+'.pkl'))

def build_models():
    directory=os.path.join(DIR_KOGAN,'port_ret','eq')
    names=[fn[:-4] for fn in os.listdir(directory)]
    names_list=list(itertools.combinations(names,2))+list(itertools.combinations(names,3))
    multi_task(_generate_models, names_list)


def get_pca_model(n=3):
    directory=os.path.join(DIR_KOGAN,'port_ret','eq')
    fns=os.listdir(directory)

    dfs=[]
    for fn in fns:
        df=pd.read_pickle(os.path.join(directory,fn))['tb']
        dfs.append(df)

    comb=pd.concat(dfs,axis=1,keys=[fn[:-4] for fn in fns])
    comb=comb.dropna(thresh=int(comb.shape[1]*0.8))
    comb=comb.fillna(0)

    X=comb.values
    pca=PCA(n_components=n)
    return pd.DataFrame(pca.fit_transform(X),index=comb.index,
                        columns=['pca{}'.format(i) for i in range(1,n+1)])







# if __name__ == '__main__':
#     build_models()













