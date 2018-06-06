# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  13:25
# NAME:FT-download_from_ftresearch.py

import pymysql
import pandas as pd
import os
import numpy as np
from config import DRAW, DCSV, DPKL
from data.dataApi import read_raw
from tools import number2dateStr


def download_from_server(tbname,database='filesync'):
    try:
        db = pymysql.connect('192.168.1.140', 'ftresearch', 'FTResearch',
                             database,charset='utf8')
    except:
        db=pymysql.connect('localhost','root','root',database,charset='utf8')
    cur = db.cursor()
    query = 'SELECT * FROM {}'.format(tbname)
    cur.execute(query)
    table = cur.fetchall()
    cur.close()
    table = pd.DataFrame(list(table),columns=[c[0] for c in cur.description])
    table.to_csv(os.path.join(DRAW, '{}.csv'.format(tbname)))


#--------------------------- download ftresearch------------------------------------------
def download_ftresearch():
    tbnames = [
        'equity_cash_dividend',
        'equity_consensus_forecast',
        'equity_fundamental_info',
        'equity_selected_balance_sheet',
        'equity_selected_cashflow_sheet',
        'equity_selected_cashflow_sheet_q',
        'equity_selected_income_sheet',
        'equity_selected_income_sheet_q',
        'equity_selected_indice_ir',
        'equity_selected_trading_data',
        'equity_shareholder_big10',
        'equity_shareholder_float_big10',
        'equity_shareholder_number',
    ]

    for tbname in tbnames:
        download_from_server(tbname,database='ftresearch')
        print(tbname)

#-------------------------------download filesync-------------------------------
def download_filesync():
    tbnames=['ashareindicator','asharecalendar']
    for tbname in tbnames:
        download_from_server(tbname,'filesync')










#TODO:check cashflow and income sheet

#TODO: there is some invalid codes at the end of the index
#TODO:just use these tables to calculate new indicators



