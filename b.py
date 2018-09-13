# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-11  17:16
# NAME:FT_hp-b.py



from config import SINGLE_D_INDICATOR
from data.dataApi import read_raw, read_local
import pandas as pd
import os


FORWARD_LIMIT=20
SMOOTH_PERIOD=30



def save_indicator(df,name):
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))


con = read_raw('equity_consensus_forecast')
    # con = read_raw('equity_consensus_forecast')

con['trd_dt'] = pd.to_datetime(con['trd_dt'])
r=con.groupby('trd_dt').apply(lambda df:df.shape[0])

con.shape