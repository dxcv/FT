# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-30  16:55
# NAME:FT_hp-4 prune.py

from empirical.config_ep import DIR_DM, DIR_DM_NORMALIZED, PERIOD_THRESH
import os
import pandas as pd
from tools import multi_process
import pickle



def _read_s(fn):
    s=pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',fn))['tb']
    s.name=fn[:-4]
    return s

def get_all_spread():# long-short return
    fns = os.listdir(os.path.join(DIR_DM, 'port_ret', 'eq'))
    df = pd.concat(multi_process(_read_s, fns, n=20), axis=1)
    df=df.dropna(axis=1,thresh=PERIOD_THRESH)#trick: prune
    df.to_pickle(os.path.join(DIR_DM,'spread.pkl'))

def get_playing_indicators():
    df=pd.read_pickle(os.path.join(DIR_DM,'spread.pkl'))
    indicators1=set(df.columns)
    indicators2=set([fn[:-4] for fn in os.listdir(DIR_DM_NORMALIZED)])
    indicators=list(indicators1.intersection(indicators2))

    pickle.dump(indicators,open(os.path.join(DIR_DM,'playing_indicators.pkl'),'wb'))



if __name__ == '__main__':
    get_all_spread()
    get_playing_indicators()
