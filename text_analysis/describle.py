# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-06  08:58
# NAME:FT-describle.py
import os
import random
import re
import pandas as pd
from data.dataApi import read_local_pkl,read_raw
from data.database_api import database_api as dbi

path=r'D:\zht\database\quantDb\internship\FT\text_analysis\analysis'

def get_name_map():
    fdmt=read_raw('equity_fundamental_info')
    names=fdmt[['stkcd','stkname']]
    # names=dbi.get_stocks_data('equity_fundamental_info',['stkcd','stkname'])
    test=names[~names.duplicated(keep='first')].dropna()
    name_stkcd={s.values[1]:s.values[0] for _,s in test.iterrows()}
    return name_stkcd

def parse_txt():
    drct = r'D:\zht\database\quantDb\internship\FT\text_analysis\result'
    fns=os.listdir(drct)
    items=[]
    for fn in fns:
        content=open(os.path.join(drct,fn),encoding='utf8').read()
        date=re.findall(r'\d{4}.\d{2}.\d{2}',content)[0]
        date=date.replace(r'年', '.').replace(r'月', '.')
        stocks=re.findall(r'\S*\s:\s\d*',content)
        items.append((date,stocks,fn))

    data = []
    for item in items:
        info_date = item[0]
        for ele in item[1]:
            stkname, mark = [i.strip() for i in ele.split(':')]
            data.append([info_date, stkname, mark,item[2]])
    signal = pd.DataFrame(data, columns=['info_date', 'stkname', 'mark','fn'])
    signal['info_date'] = pd.to_datetime(signal['info_date'])
    signal['mark'] = signal['mark'].map(int)
    signal['stkcd']=signal['stkname'].map(get_name_map())
    newcols=['info_date','stkcd','mark','stkname','fn']
    signal=signal[newcols]
    signal=signal.sort_values(['info_date','stkcd'])
    signal=signal.dropna().reset_index(drop=True)
    return signal

def _add_trd_dt(df):
    calendar=read_raw('asharecalendar')
    trd_dt=pd.to_datetime(calendar['TRADE_DAYS'].map(str)).drop_duplicates().sort_values()
    df['trd_dt']=df['info_date'].map(lambda x:trd_dt.values[trd_dt.searchsorted(x)[0]])
    return df

def get_signal():
    signal=parse_txt()
    signal=_add_trd_dt(signal)
    signal.to_pickle(r'e:\a\signal.pkl')

trading=read_local_pkl('equity_selected_trading_data')
ret=trading['pctchange']
ret=ret.unstack(level='stkcd')/100

# ret=trading['adjclose'].unstack(level='stkcd').pct_change()
signal=pd.read_pickle(r'e:\a\signal.pkl')
name_stkcd=get_name_map()
signal['stkcd']=random.choices(population=list(name_stkcd.values()),k=signal.shape[0])
# signal['trd_dt']=random.choices(population=ret.index.tolist()[:-300],k=signal.shape[0])
signal=signal.sort_values(['trd_dt','stkcd'])
signal=signal[~signal.duplicated(['trd_dt','stkcd'],keep='last')]
bench=dbi.get_index_data(['sz50','hs300','zz500']).pct_change()

def get_zz500_window(window=200):
    event_dates=signal['trd_dt'].unique()
    dates = ret.index.tolist()
    ss=[]
    for ed in event_dates:
        start = dates[dates.index(ed) - window]
        try:
            end = dates[dates.index(ed) + window]
        except:
            end=dates[-1]
        s=bench.loc[start:end,'zz500']
        s=s.reset_index(drop=True)
        s.index=[ind-window for ind in s.index]
        ss.append(s)
    zz500_window=pd.concat(ss,axis=1).mean(axis=1)
    fig=zz500_window.cumsum().plot().get_figure()
    fig.savefig(os.path.join(path,'zz500_window.png'))

# get_zz500_window()



def get_pnl(window=10,weight=True):
    if weight:
        signal_groups=signal.groupby('trd_dt')
        dates=ret.index.tolist()
        pnls=[]
        for td,g in list(signal_groups):
            start=td
            try:
                end=dates[dates.index(td)+window]
            except:
                end=dates[-1]
            subdf=ret.loc[start:end,g['stkcd']]
            weights={g['stkcd'].values[i]:g['mark'].values[i] for i in range(g.shape[0])}
            subdf=subdf.dropna(axis=1,how='all')

            for stkcd in subdf.columns:
                subdf[stkcd] = subdf[stkcd] * weights[stkcd]
            pnl=subdf.sum(axis=1)/sum([weights[c] for c in subdf.columns])
            pnls.append(pnl)
        result=pd.concat(pnls,axis=1).mean(axis=1)
        result.name='pnl'
    else:
        signal_groups = signal.groupby('trd_dt')
        df = pd.DataFrame(index=ret.index, columns=ret.columns)
        dates = df.index.tolist()
        pnls = []
        for td, g in list(signal_groups):
            start = td
            try:
                end = dates[dates.index(td) + window]
            except:
                end = dates[-1]
            subdf = ret.loc[start:end, g['stkcd']]
            subdf = subdf.dropna(axis=1, how='all')
            pnl = subdf.mean(axis=1)
            pnls.append(pnl)
        result = pd.concat(pnls, axis=1).mean(axis=1)
        result.name = 'pnl'
    return result

def get_raw(weight=True,bm='zz500'):
    windows = [1, 3, 10, 30, 50, 100, 200]
    pnls = []
    for window in windows:
        pnls.append(get_pnl(window,weight))
        print(window)

    comb = pd.concat(pnls + [bench[bm]], axis=1, keys=windows + [bm])
    comb = comb[comb.index >= pnls[0].index[0]]
    comb=comb.fillna(0)
    fig=(comb+1).cumprod().plot().get_figure()
    fig.savefig(os.path.join(path,'raw{}.png'.format('_weighted' if weight else '')))

def get_relative(weight=True,bm='zz500'):
    windows=[1,3,10,30,50,100,200]
    pnls=[]
    for window in windows:
        pnls.append(get_pnl(window,weight))
        print(window)

    comb=pd.concat(pnls + [bench[bm]], axis=1, keys=windows + [bm])
    comb=comb[comb.index>=pnls[0].index[0]]
    relative=comb[windows].apply(lambda s:s-comb[bm])
    relative=relative.fillna(0)
    fig=(relative+1).cumprod().plot().get_figure()
    fig.savefig(os.path.join(path,'relative{}.png'.format('_weighted' if weight else '')))

#event study
def event_method(window,weight=True,bm='zz500'):
    signal_groups = signal.groupby('trd_dt')
    dates = ret.index.tolist()
    relatives = []
    for td, g in list(signal_groups):
        print(td)
        start = dates[dates.index(td) - window]
        try:
            end = dates[dates.index(td) + window]
        except:
            end = dates[-1]
        subdf = ret.loc[start:end, g['stkcd']]
        subdf = subdf.dropna(axis=1, how='all')
        if weight:
            weights = {g['stkcd'].values[i]: g['mark'].values[i] for i in
                       range(g.shape[0])}
            for stkcd in subdf.columns:
                subdf[stkcd] = subdf[stkcd] * weights[stkcd]

            pnl = subdf.sum(axis=1) / sum([weights[c] for c in subdf.columns])
            pnl.name = 'pnl'
        else:
            pnl=subdf.mean(axis=1)
            pnl.name='pnl'

        comb = pd.concat([pnl, bench[bm]], axis=1, join='inner')
        relative = comb['pnl'] - comb[bm]
        relative = relative.reset_index(drop=True)
        relative.index = [ind - window for ind in relative.index]
        # relative.index=range(-window,window+1)
        relatives.append(relative)
    result = pd.concat(relatives, axis=1)
    return result.mean(axis=1)

def event_analysis():
    window=200
    r2=event_method(window,weight=False)
    r1=event_method(window,weight=True)
    df=pd.concat([r1,r2],axis=1,keys=['weighted','equal'])
    fig_cs=(1+df).cumprod().plot().get_figure()
    fig_cs.show()
    fig_cs.savefig(os.path.join(path,'event_study_weighted_{}_cumprod.png'.format(window)))

    fig_cs=df.cumsum().plot().get_figure()
    fig_cs.savefig(os.path.join(path,'event_study_weighted_{}_cumsum.png'.format(window)))


def run():
    get_raw(True)
    get_raw(False)
    get_relative(True)
    get_relative(False)
    event_analysis()

