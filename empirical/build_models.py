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

from config import  DIR_KOGAN


def generate_models(names):
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
    multiprocessing.Pool(30).map(generate_models,names_list)

if __name__ == '__main__':
    build_models()











