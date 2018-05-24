# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  13:25
# NAME:FT-download_all_tables.py
import pymysql
import pandas as pd
import os
import numpy as np


from config import LOCAL_DB

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

def connect_database():
    db = pymysql.connect('192.168.1.140', 'ftresearch', 'FTResearch',
                         'ftresearch')
    cur = db.cursor()
    return cur

def download_all_tables():
    dateFields = ['report_period', 'trd_dt', 'ann_dt', 'holder_enddate',
                  'listdate']
    cur = connect_database()
    for tbname in tbnames:
        query = 'SELECT * FROM {}'.format(tbname)
        cur.execute(query)
        table = cur.fetchall()
        table = pd.DataFrame(list(table),columns=[c[0] for c in cur.description])
        #convert the data format of date fields
        for dfd in dateFields:
            if dfd in table.columns:
                # table[dfd]=table[dfd].map(
                #     lambda x:x[:4]+'-'+x[4:6]+'-'+x[6:] if x else np.nan)
                table[dfd]=pd.to_datetime(table[dfd])
        del table['create_time']
        del table['update_time']
        # with pd.HDFStore(os.path.join(LOCAL_DB,tbname+'.h5')) as file:
        #     file.put(tbname, table)
        table.to_pickle(os.path.join(LOCAL_DB,'{}.pkl'.format(tbname)))
        table.to_csv(os.path.join(LOCAL_DB,'{}.csv'.format(tbname)))
        print(tbname)

if __name__ == '__main__':
    download_all_tables()
