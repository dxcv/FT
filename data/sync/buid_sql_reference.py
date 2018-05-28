# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-27  00:14
# NAME:FT-buid_sql_reference.py
import os
from io import StringIO
import pymysql
import pandas as pd
import xlsxwriter

directory=r'D:\zht\database\quantDb\internship\FT\documents\filesync_info'


def download_table_info():
    dir_csv = r'E:\FT_Users\HTZhang\filesync_info\csv'
    dir_html = r'E:\FT_Users\HTZhang\filesync_info\html'

    db=pymysql.connect('192.168.1.140','ftresearch','FTResearch','filesync',charset='utf8')
    cur=db.cursor()
    q_showtables='show tables'

    cur.execute(q_showtables)
    tables=cur.fetchall()
    tables=[t[0] for t in tables]
    for i,table in enumerate(tables):
        query="select * from information_schema.columns where table_name='{}'".format(table)
        cur.execute(query)
        info=pd.DataFrame(list(cur.fetchall()))
        info=info[[2,3,15,16,19]]
        info.columns=['table','field','format','pri','comment']
        info.to_csv(os.path.join(dir_csv,table+'.csv'),encoding='gbk')
        info.to_csv(os.path.join(dir_html,table+'.html'),encoding='gbk')
        print(i,table)

def build_xlsx():
    excel=os.path.join(directory,'目录.xlsx')
    df=pd.read_excel(excel,header=None)
    df.columns=['c1','c2','table']
    df[['c1','c2']]=df[['c1','c2']].ffill()
    df=df.dropna()
    df=df.reset_index(drop=True)

    def map_link(s):
        hp=r'external:csv\{}.csv'.format(s.split('(')[0].lower())
        return hp

    workbook=xlsxwriter.Workbook(os.path.join(directory,'reference.xlsx'))
    worksheet=workbook.add_worksheet('reference')
    # Format the first column
    worksheet.set_column('A:A', 30)
    red_format = workbook.add_format({
        'font_color': 'red',
        'bold':       1,
        'underline':  1,
        'font_size':  12,
    })

    i=0
    for _,s in df.iterrows():
        i+=1
        worksheet.write_string('A{}'.format(i),s['c1'])
        worksheet.write_string('B{}'.format(i),s['c2'])
        worksheet.write_url('C{}'.format(i),map_link(s['table']),
                            string=s['table'],
                            cell_format=red_format)
    workbook.close()


def combine_all_reference():
    excel=os.path.join(directory,'目录.xlsx')
    df=pd.read_excel(excel,header=None)
    df.columns=['c1','c2','table']
    df[['c1','c2']]=df[['c1','c2']].ffill()
    df=df.dropna()
    df=df.reset_index(drop=True)
    df['csvName']=df['table'].map(lambda x:x.split('(')[0].lower())

    path=os.path.join(directory,'csv')
    fns=os.listdir(path)

    infos=[]
    for fn in fns:
        info=pd.read_csv(os.path.join(path,fn),index_col=0,encoding='gbk')
        index=df.index[df['csvName']==fn[:-4]].tolist()
        if index:
            index=index[0]
            c1=df.at[index,'c1']
            c2=df.at[index,'c2']
            table=df.at[index,'table']
            info['c1']=c1
            info['c2']=c2
            info['table']=table
        else:
            info['c1']=None
            info['c2']=None
            info['table']=None
            info.fillna('Missing')
            print(fn)
        infos.append(info)
    comb=pd.concat(infos,axis=0)
    comb['field']=comb['field'].str.lower()
    newOrder=['c1','c2','table','field','format','pri','comment']
    comb=comb[newOrder]
    comb=comb.sort_values(['c1','c2','table'])
    comb.to_csv(os.path.join(directory,'combined.csv'),encoding='gbk')







