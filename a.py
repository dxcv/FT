# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-27  21:42
# NAME:FT_hp-a.py

import os
from itertools import combinations

import pandas as pd
# import copy
import matplotlib.pyplot as plt
from empirical.config_ep import DIR_BASEDATA

fns = os.listdir(os.path.join(DIR_BASEDATA, 'normalized_conditional'))

df=pd.read_pickle(os.path.join(DIR_BASEDATA,'normalized_conditional',fns[0]))