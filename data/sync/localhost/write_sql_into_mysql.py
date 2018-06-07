# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-06  08:45
# NAME:FT-write_sql_into_mysql.py
import time
from os import system
import os

from tools import monitor


def run_sql(fp):
    username='root'
    password='root'
    host='localhost'
    port=3306

    command="""mysql -u %s -p"%s" --host %s --port %s < %s"""%(username, password, host, port, fp)
    system(command)
    print(fp)


def run():
    directory=r'E:\filesync'
    fps=[os.path.join(directory,fn) for fn in os.listdir(directory)]
    fps=sorted(fps,key=os.path.getsize)

    with open(r'e:\write_sql_into_mysql.log','w') as f:
        for fp in fps:
            run_sql(fp)
            info='{}-> {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'),fp)
            f.write(info)
            print(info)

if __name__ == '__main__':
    run()
