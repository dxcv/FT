# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  10:05
# NAME:FT_hp-get_basedata.py
import pandas as pd
import os
from empirical.config import DIR_KOGAN

# rpM.pkl is copied from D:\zht\database\quantDb\researchTopics\assetPricing2_new\data\pkl_unfiltered
rpM=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata','rpM.pkl'))


#benchmark models are all copied from D:\zht\database\quantDb\researchTopics\assetPricing2_new\data\pkl_filtered

def get_benchmark(name):
    df=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata',name+'.pkl'))
    return df



