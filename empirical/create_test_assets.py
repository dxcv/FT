# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  10:48
# NAME:FT_hp-create_test_assets.py
import multiprocessing
import os

import itertools
import pandas as pd
from empirical.config_ep import DIR_KOGAN


def create_assets():
    directory = os.path.join(DIR_KOGAN, 'port_ret', 'eq')
    fns=os.listdir(directory)
    for fn in fns:
        df=pd.read_pickle(os.path.join(directory,fn)).drop('tb',axis=1)
        # fixme:create eret,rather than the raw return
        df.to_pickle(os.path.join(DIR_KOGAN,'assets','eq',fn))
        print(fn)

if __name__ == '__main__':
    create_assets()


