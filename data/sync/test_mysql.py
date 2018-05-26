# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-27  00:14
# NAME:FT-test_mysql.py
import os

import pymysql
import pandas as pd

dirInfo=r'E:\FT_Users\HTZhang\ftresearch_info'

db=pymysql.connect('192.168.1.140','ftresearch','FTResearch','barra')
cur=db.cursor()
q_showtables='show tables'

cur.execute(q_showtables)
tables=cur.fetchall()
tables=[t[0] for t in tables]
for table in tables:
    query='describe {}'.format(table)
    cur.execute(query)
    info=pd.DataFrame(list(cur.fetchall()))
    info.to_csv(os.path.join(dirInfo,table+'.csv'))

