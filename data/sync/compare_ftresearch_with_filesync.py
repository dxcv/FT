# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-28  16:30
# NAME:FT-compare_ftresearch_with_filesync.py
import pandas as pd


def parse_balancesheet():
    '''
    balancesheet
    bs=bs[bs['statement'].isin(['408001000','408005000'])]
    if there is 408005000 (合并报表（更正前)),keep 408005000 and delete 408001000
    '''
    bs = pd.read_csv(r'e:\asharebalancesheet.csv', sep='\t', header=None)
    info=pd.read_csv(r'D:\zht\database\quantDb\internship\FT\documents\filesync_info\csv\asharebalancesheet.csv',encoding='gbk')
    bs.columns=info['field'].str.lower()
    bs = bs[bs['statement_type'].isin([408001000, 408005000])]
    '''
    there are some duplicates even after controlling ['wind_code','report_period','satement_type']
    we keep the item with the earliest ann_dt
    '''
    bs=bs.sort_values(['wind_code','report_period','statement_type','ann_dt'],ascending=True)
    bs=bs[~bs.duplicated(subset=['wind_code','report_period','statement_type'],keep='first')]
    '''
    after controlling ['wind_code','report_period'],if there is any duplicate,
    keep the one with statement_type code as 408005000 and delete the item with
    statement_type code as 408001000.Since df has sorted on statement_type,we
    only need to keep the last one.
    '''
    bs=bs[~bs.duplicated(subset=['wind_code','report_period'],keep='last')]
    #TODO: what's "opdate" and "opmode"?
    return bs







