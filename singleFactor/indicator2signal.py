# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-16  09:37
# NAME:FT_hp-indicator2signal.py



import multiprocessing

from config import SINGLE_D_INDICATOR, DIR_SIGNAL
import os
import pandas as pd
from singleFactor.singleTools import convert_indicator_to_signal
from tools import multi_process


def indicator2signal(name):
    try:
        df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
        signal = convert_indicator_to_signal(df, name)
        signal.to_pickle(os.path.join(DIR_SIGNAL, name + '.pkl'))
        print(name)
    except:
        print('wrong!----------->{}'.format(name))

def run():
    fns=os.listdir(SINGLE_D_INDICATOR)
    names=[fn[:-4] for fn in fns]
    print('total number: {}'.format(len(names)))
    checked=[fn[:-4] for fn in os.listdir(DIR_SIGNAL)]
    names=[n for n in names if n not in checked]
    print('unchecked: {}'.format(len(names)))
    multi_process(indicator2signal,names,5)
    for name in names:
        indicator2signal(name)

    # pool=multiprocessing.Pool(1)
    # pool.map(indicator2signal,names)

if __name__ == '__main__':
    run()


# df=pd.read_pickle(os.path.join(SINGLE_D_INDICATOR,'T__idioVol_30.pkl'))
