# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-07  17:11
# NAME:FT-backtest_cz.py
import multiprocessing

from config import SINGLE_D_INDICATOR, DIR_SIGNAL, DIR_BACKTEST_RESULT, \
    LEAST_CROSS_SAMPLE
import os
import pandas as pd
from data.dataApi import read_local
from backtest.main import quick
from tools import clean


def test_one(name):
    print(name)
    df = pd.read_pickle(os.path.join(SINGLE_D_INDICATOR, name + '.pkl'))
    df=df.stack().to_frame().swaplevel().sort_index()
    df.columns=[name]
    fdmt = read_local('equity_fundamental_info')
    data=pd.concat([fdmt,df],axis=1,join='inner')

    data=data.dropna(subset=['type_st','young_1year'])
    data = data[(~data['type_st']) & (~ data['young_1year'])]  # 剔除st 和上市不满一年的数据
    data=data.dropna(subset=['wind_indcd',name])
    data=data.groupby('trd_dt').filter(lambda x:x.shape[0]>LEAST_CROSS_SAMPLE)


    cleaned_data=clean(data,col=name,by='trd_dt')
    signal=pd.pivot_table(cleaned_data,values=name,index='trd_dt',columns='stkcd').sort_index()
    signal=signal.shift(1)#trick:

    directory=os.path.join(DIR_BACKTEST_RESULT,name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    signal.to_csv(os.path.join(directory, 'signal.csv'))

    # directory=os.path.join(DIR_BACKTEST_RESULT,name)
    # signal=pd.read_csv(os.path.join(directory,'signal.csv'),index_col=0,parse_dates=True)

    start='2010'
    results,fig=quick(signal,name,start=start)

    fig.savefig(os.path.join(directory,name+'.png'))
    for k in results.keys():
        results[k].to_csv(os.path.join(directory,k+'.csv'))


def test_all():
    # path = 'indicators.xlsx'
    # df = pd.read_excel(path, sheet_name='valid')
    # names=df['name']
    fns=os.listdir(SINGLE_D_INDICATOR)
    names=[fn[:-4] for fn in fns]
    print(len(names))
    checked=[fn for fn in os.listdir(DIR_BACKTEST_RESULT)]
    names=[n for n in names if n not in checked]
    print(len(names))
    # pool = multiprocessing.Pool(4)
    # pool.map(test_one, names)

    for i,name in enumerate(names):
        try:
            test_one(name)
            print(i,name)
        except:
            pass

def debug():
    name='Q__roe'
    directory=os.path.join(DIR_BACKTEST_RESULT,name)
    signal=pd.read_csv(os.path.join(directory,'signal.csv'),index_col=0)
    start = '2010'
    end = '2016'
    results,fig=quick(signal,name,start,end)
    fig.savefig(os.path.join(directory,name+'.png'))
    for k in results.keys():
        results[k].to_csv(os.path.join(directory,k+'.csv'))

if __name__ == '__main__':
    test_all()



