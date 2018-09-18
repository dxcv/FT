# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-18  09:31
# NAME:FT_hp-dm_api.py
import pandas as pd
import os
import pickle

from empirical.config_ep import DIR_DM_GTA, PERIOD_THRESH
from tools import multi_process


def _read_s(name):
    s=pd.read_pickle(os.path.join(DIR_DM_GTA,'port_ret','eq', f'{name}.pkl'))['tb']
    s.name= name
    return s

def get_playing_indicators_and_raw_factors():
    inds1=os.listdir(os.path.join(DIR_DM_GTA,'port_ret','eq'))
    inds2=os.listdir(os.path.join(DIR_DM_GTA,'normalized'))
    inds1=[ind[:-4] for ind in inds1]
    inds2=[ind[:-4] for ind in inds2]

    indicators=list(set(inds1).intersection(set(inds2)))

    df = pd.concat(multi_process(_read_s, indicators, n=20), axis=1,sort=True)
    df=df.dropna(axis=0,thresh=int(df.shape[1]*0.8)) #trick: delete those months with too few of factors
    df = df.dropna(axis=1, thresh=PERIOD_THRESH)  # trick: to enter into our sample, a factor must cover at least n months in our given period
    df = df.fillna(0)  # trick: fillna with 0 to facilitate the following calculation

    df.to_pickle(os.path.join(DIR_DM_GTA,'raw_factors.pkl'))

    playing_indicators=df.columns.tolist()
    pickle.dump(playing_indicators,open(os.path.join(DIR_DM_GTA,'playing_indicators.pkl'),'wb'))


def get_playing_indicators():
    path=os.path.join(DIR_DM_GTA,'playing_indicators.pkl')
    return pickle.load(open(path,'rb'))

def get_raw_factors():
    path=os.path.join(DIR_DM_GTA,'raw_factors.pkl')
    return pd.read_pickle(path)

factors=get_raw_factors()
factors.shape


df=pd.read_pickle(r'G:\FT_Users\HTZhang\empirical\data_mining\based_on_gta\port_ret\eq\ratio_x_chg_over_lag_y-A001101000-A003101000.pkl')

def main():
    get_playing_indicators_and_raw_factors()

# if __name__ == '__main__':
#     main()
