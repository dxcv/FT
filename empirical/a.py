# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-31  20:19
# NAME:FT_hp-a.py



import pandas as pd
import numpy as np


df=pd.DataFrame(np.random.random((3,5)),index=['a','b','c'])



df.apply(lambda s:s[[s.name.split('_')[0],s.name.split('_')[-1][:-4]]].reindex(s.index))


s=pd.Series(range(3),index=['a','b','c'])

s[['a','b']].reindex(s.index)

