# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-27  21:42
# NAME:FT_hp-a.py
import pandas as pd
import os

from config import DIR_TMP

df=pd.read_pickle(os.path.join(DIR_TMP,'grade_strategy__M_100.pkl'))
