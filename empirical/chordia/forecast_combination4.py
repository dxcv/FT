# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-05  11:32
# NAME:FT_hp-forecast_combination4.py
import os
import pandas as pd


from empirical.chordia_and_yan.aggregation_with_fm3 import fm_predict, \
    tmb_with_fm_predicted
from empirical.chordia_and_yan.identify_anomalies1 import get_prominent_indicators
from empirical.config_ep import DIR_DM_NORMALIZED, DIR_CHORDIA
from tools import z_score, multi_process
import matplotlib.pyplot as plt


def forecast_combination(smooth_period=60):
    '''
    combination forecast with all the indicators returned from the function get_prominent_indicators()
    Args:
        smooth_period:

    Returns:

    '''
    inds = get_prominent_indicators()
    ss=[]
    for ind in inds:
        indicator = pd.read_pickle(
            os.path.join(DIR_DM_NORMALIZED, ind + '.pkl')).stack()
        indicator.name = ind
        indicator = indicator.to_frame()
        indicator = indicator.dropna()
        predicted=fm_predict(indicator, smooth_period)
        predicted.index.names=['month_end','stkcd']
        predicted=predicted.groupby('month_end').apply(z_score)
        ss.append(predicted)

    tmbs=[]
    for n in [1,3,5,10,15,len(ss)]:
        p=pd.concat(ss[:n],axis=1).sum(axis=1)
        p.name='predicted'
        tmb=tmb_with_fm_predicted(p)
        tmb.name=n
        tmbs.append(tmb)
    df=pd.concat(tmbs,axis=1)
    df.cumsum().plot()
    plt.savefig(os.path.join(DIR_CHORDIA,
                             f'combination_forecast_{smooth_period}.png'))

def travese_smooth_period():
    smooth_periods=[1,6,12,36,60]
    multi_process(forecast_combination,smooth_periods,n=5)

def fm_predict_with_one_indicator(ind):
    indicator = pd.read_pickle(
        os.path.join(DIR_DM_NORMALIZED, ind + '.pkl')).stack()
    indicator.name = ind
    indicator = indicator.to_frame()
    indicator = indicator.dropna()
    ss = []
    for i in [1, 6, 12, 36,60]:
        predicted=fm_predict(indicator,i)
        s=tmb_with_fm_predicted(predicted)
        s.name = i
        ss.append(s)

    df = pd.concat(ss, axis=1)

    df.cumsum().plot()
    plt.savefig(os.path.join(DIR_CHORDIA, f'{ind}.png'))
    plt.close()
    print(ind)



def traverse_indicators():
    inds = get_prominent_indicators()
    multi_process(fm_predict_with_one_indicator, inds, 10)

def main():
    travese_smooth_period()
    traverse_indicators()


if __name__ == '__main__':
    main()
