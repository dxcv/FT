# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-30  15:55
# NAME:FT_hp-a.py
from empirical.config_ep import DIR_DM
import os
import pandas as pd
from tools import multi_thread, multi_process
import time



def _read_s(fn):
    s=pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',fn))['tb']
    s.name=fn[:-4]
    return s

def get_all_spread():# long-short return
    fns = os.listdir(os.path.join(DIR_DM, 'port_ret', 'eq'))
    df = pd.concat(multi_process(_read_s, fns, n=20), axis=1)
    df.to_pickle(os.path.join(DIR_DM,'spread.pkl'))


# if __name__ == '__main__':
#     get_all_spread()


port_ret=pd.read_pickle(os.path.join(DIR_DM,'spread.pkl'))
port_ret=port_ret.dropna(axis=1,thresh=len(port_ret)*0.6)
