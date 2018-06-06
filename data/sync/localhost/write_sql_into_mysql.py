# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-06  08:45
# NAME:FT-write_sql_into_mysql.py

import MySQLdb as mdb

import datetime, time


def run_sql_file(filename, connection):
    '''
    The function takes a filename and a connection as input
    and will run the SQL query on the given connection
    '''
    start = time.time()

    file = open(filename, 'r')
    sql = s = " ".join(file.readlines())
    print
    "Start executing: " + filename + " at " + str(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + "\n" + sql
    cursor = connection.cursor()
    cursor.execute(sql)
    connection.commit()

    end = time.time()
    print
    "Time elapsed to run the query:"
    print
    str((end - start) * 1000) + ' ms'


def main():
    connection = mdb.connect('127.0.0.1', 'root', 'password', 'database_name')
    run_sql_file("my_query_file.sql", connection)
    connection.close()


if __name__ == "__main__":
    main()

