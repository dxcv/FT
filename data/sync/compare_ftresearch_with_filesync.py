# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-28  16:30
# NAME:FT-compare_ftresearch_with_filesync.py
import pandas as pd


bs=pd.read_csv(r'e:\asharebalancesheet.csv',sep='\t',header=None)
info=pd.read_csv(r'D:\zht\database\quantDb\internship\FT\documents\filesync_info\csv\asharebalancesheet.csv',encoding='gbk')
bs.columns=info['field'].str.lower()

bs_ft=pd.read_csv(r'D:\zht\database\quantDb\internship\FT\database\equity_selected_balance_sheet.csv')


bs['statement_type'].value_counts()


bs=bs.sort_values(['wind_code','report_period','ann_dt'])

bs_ft=bs_ft.sort_values(['stkcd','report_period','ann_dt'])

bs=bs[bs['statement_type'].isin(['408001000','408005000','408050000'])]

dup=bs[bs.duplicated(subset=['wind_code','report_period'])]

dup[['wind_code','report_period']].head(10)

stocks=bs['wind_code'].unique()
for stock in stocks:
    a=bs[bs['wind_code']==stock]['statement_type'].value_counts().shape[0]
    if a>2:
        print(stock,a)




bs[bs['wind_code']=='000023.SZ'].to_csv(r'e:\a\bs.csv')
bs_ft[bs_ft['stkcd']=='000023.SZ'].to_csv(r'e:\a\bs_ft.csv')



# balancesheet
# bs=bs[bs['statement'].isin(['408001000','408005000'])]
# if there is 408001000,keep 408001000 and delete 408005000


#408001000 合并报表
#!408004000

# if there is 408005000 (合并报表（更正前)),use 40800500 rather than 408001000


#TODO:check cashflow and income sheet
