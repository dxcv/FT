# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-13  17:54
# NAME:FT_hp-_indicator2signal.py
import multiprocessing

from config import SINGLE_D_INDICATOR, DIR_SIGNAL,DIR_DM_RESULT, DIR_DM_SIGNAL, DIR_DM
import os
import pandas as pd
from singleFactor.singleTools import convert_indicator_to_signal


def indicator2signal_dm(name):
    try:
        df = pd.read_pickle(os.path.join(DIR_DM_RESULT, name, 'daily.pkl'))
        signal = convert_indicator_to_signal(df, name)
        signal.to_pickle(os.path.join(DIR_DM_SIGNAL, name + '.pkl'))
        print(name)
    except:
        with open(os.path.join(DIR_DM,'indicator2signal_failed.txt'),'a') as f:
            f.write(name+'\n')
        print('failed!----------->{}'.format(name))

def run_dm():
    names=os.listdir(DIR_DM_RESULT)
    print('total number: {}'.format(len(names)))
    checked=[fn[:-4] for fn in os.listdir(DIR_DM_SIGNAL)]
    with open(os.path.join(DIR_DM,'indicator2signal_failed.txt')) as f:
        failed=f.read().split('\n')
    names=[n for n in names if n not in checked+failed]
    print('unchecked: {}'.format(len(names)))
    pool = multiprocessing.Pool(10)
    pool.map(indicator2signal_dm, names)

def debug():
    name='T__beta_30'
    df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
    signal=convert_indicator_to_signal(df,name)
    signal.to_pickle(os.path.join(DIR_SIGNAL, name + '.pkl'))

if __name__ == '__main__':
    # run()
    run_dm()




#TODO: revert the negative signal
