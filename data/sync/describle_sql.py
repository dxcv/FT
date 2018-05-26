# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-26  11:50
# NAME:FT-describle_sql.py


import pandas as pd

df=pd.read_csv(r'e:\a\mysql_info.csv')

def _convert(s):
    print(s)
    if s.endswith('K'):
        return float(s[:-1])*1024
    elif s.endswith('M'):
        return float(s[:-1])*1024*1024
    elif s.endswith('G'):
        return float(s[:-1])*1024*1024*1024
    else:
        print('----------->wrong')



df['bytes']=df['total size'].apply(_convert)
df['M']=df['bytes']/(1024*1024)
df['G']=df['M']/1024
df=df.sort_values('G',ascending=False)

df['G'].sum()
