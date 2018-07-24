# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-21  11:06
# NAME:FT_hp-a.py
import pandas as pd
import numpy as np
from tools import myroll


def _cal_beta(df, min_periods):
    df=df.dropna(thresh=min_periods, axis=1)
    df=df.fillna(df.mean()) #Trick: fillna with average
    # df=df.fillna(0)
    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    return pd.Series(b[1],index= df.columns[1:], name='beta')


window=12
min_periods=10

comb=pd.read_pickle(r'F:\FT_Users\HTZhang\comb.pkl')
results=myroll(comb, window).apply(_cal_beta, min_periods)
