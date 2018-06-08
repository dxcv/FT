# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-06  08:45
# NAME:FT-write_sql_into_mysql.py
import time
from os import system
import os

import pymysql
import multiprocessing


def run_sql(fp):
    username='root'
    password='root'
    host='localhost'
    port=3306

    command="""mysql -u %s -p"%s" --host %s --port %s < %s"""%(username, password, host, port, fp)
    system(command)
    print(fp)

# if __name__ == '__main__':
#     run()

def get_existed_tables():
    db = pymysql.connect('localhost', 'root', 'root', 'filesync',
                         charset='utf8')
    cur = db.cursor()
    q_showtables = 'show tables'
    cur.execute(q_showtables)
    tables = cur.fetchall()
    cur.close()
    tables = [t[0] for t in tables]
    return tables

def get_fps():
    directory=r'E:\filesync'
    fps=[os.path.join(directory,fn) for fn in os.listdir(directory)]
    fps=sorted(fps,key=os.path.getsize)

    tables=get_existed_tables()
    fps=[fp for fp in fps if os.path.basename(fp).split('_')[1][:-4] not in tables]
    return fps

def task(fp):
    run_sql(fp)
    info='{}-> {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'),fp)
    print(info)

if __name__ == '__main__':
    fps=get_fps()
    pool=multiprocessing.Pool(4)
    pool.map(task,fps)
