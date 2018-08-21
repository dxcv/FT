# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-21  11:06
# NAME:FT_hp-a.py
import pandas as pd
import numpy as np
from data.dataApi import read_local
from tools import myroll


zz500_ret_d = read_local('equity_selected_indice_ir')['zz500_ret_d']

zz500_ret_d.to_pickle(r'G:\FT_Users\HTZhang\haitong\zz500.pkl')
