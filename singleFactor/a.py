# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-06  17:53
# NAME:FT-a.py

import pandas as pd
from config import DIR_SINGLE_BACKTEST, DIR_SIGNAL, SINGLE_D_INDICATOR
import os

from singleFactor.singleTools import convert_indicator_to_signal

name = 'V__ebitdaToCap'
df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
# signal = pd.read_pickle(os.path.join(DIR_SIGNAL, name + '.pkl'))
convert_indicator_to_signal(df, name)



