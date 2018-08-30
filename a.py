# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-27  21:42
# NAME:FT_hp-a.py

import os
import pandas as pd
# import copy
from config import DIR_DM_RESULT
from memory_profiler import profile

names = os.listdir(DIR_DM_RESULT)

items=[]
for name in names:
    path=os.path.join(DIR_DM_RESULT,name,'monthly.pkl')
    if not os.path.exists(path):
        items.append(path)

print(len(items))
print(items)
