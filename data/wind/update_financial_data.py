# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-21  15:38
# NAME:FT_hp-update_financial_data.py

import pickle

import datetime
from WindPy import w
import numpy as np
import pandas as pd
import os

DIR=r'G:\FT_Users\HTZhang\FT\database\wind'
DIR_CACHE=os.path.join(DIR,'cache')

w.start()




DATE_FORMAT='%Y-%m-%d'

def test_wind():
    print(w.isconnected())
    data=w.wss("000001.SZ,000002.SZ,000004.SZ,000005.SZ", "trade_code")
    return



def get_today(format=DATE_FORMAT):
    '''
    :return: date as string like '2018-02-01'
    '''
    today=datetime.datetime.today().strftime(format)
    return today

def get_stkcd_list(date=None):
    if date is None:
        date=get_today(DATE_FORMAT)
    path=os.path.join(DIR,'stkcd_list_{}.pkl'.format(date))
    if os.path.exists(path):
        return pickle.load(open(path,'rb'))
    else:
        codes=w.wset("SectorConstituent",u"date={};sector=全部A股".format(date)).Data[1]
        f=open(path,'wb')
        pickle.dump(codes,f)
        return codes

def get_ann_dt_df(date):
    # if date is None:
    #     date=get_today(DATE_FORMAT)
    path = os.path.join(DIR, 'ann_dt_{}.pkl'.format(date))
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        # codes = get_today_stkcd_list()
        codes=get_stkcd_list(date)
        result = w.wsd(','.join(codes), "stm_issuingdate", "ED-2Q", date,
                       "Period=Q;Days=Alldays")
        ann_dt = pd.DataFrame(result.Data, index=result.Codes,
                              columns=result.Times).T
        f = open(path, 'wb')
        pickle.dump(ann_dt, f)
        return ann_dt

def get_latest(date,prefix):
    '''
    get the latest cache before "date"
    Args:
        date:
        prefix:str,{'ann_dt','stkcd_list','data'}

    Returns:

    '''
    anns = [an for an in os.listdir(os.path.join(DIR)) if
            an.startswith(prefix) and pd.to_datetime(an[-14:-4]) < pd.to_datetime(date) and an.endswith('.pkl')]

    anns = sorted(anns, key=lambda x: pd.to_datetime(x[-14:-4]))
    return pd.read_pickle(os.path.join(DIR, anns[-1]))

def get_data_for_one_stk(stkcd, rpdate, indicators):
    path=os.path.join(DIR_CACHE,f'{rpdate}_{stkcd}.csv') #trick: do not use .pkl,sometimes it may be malformed
    if os.path.exists(path):
        s=pd.read_csv(path,index_col=0,parse_dates=True)
    else:
        data = w.wsd(stkcd, ','.join(indicators), rpdate, rpdate, "unit=1;rptType=1;Period=Q;Days=Alldays").Data
        s = pd.Series([d[0] for d in data], index=indicators)
        s.fillna(value=np.nan, inplace=True)
        s.name=stkcd
        print('Getting data for {}-----{}'.format(rpdate,stkcd))
        s.to_csv(path)#trick: cache
    return s

def get_target_df(date):
    '''
        get the df to be updated at the given date
    Returns:DataFrame, index is report period and the columns is stock codes, the
        value is the announcement date.

    Args:
        date:

    Returns:

    '''
    # if date is None:
    #     date=get_today()
    new_ann_dt=get_ann_dt_df(date)
    old_ann_dt=get_latest(date,'ann_dt')

    new_index=new_ann_dt.index.union(old_ann_dt.index)
    new_column=new_ann_dt.columns.union(old_ann_dt.columns)

    new_ann_dt=new_ann_dt.reindex(index=new_index,columns=new_column)
    old_ann_dt=old_ann_dt.reindex(index=new_index,columns=new_column)
    target_df=new_ann_dt[new_ann_dt!=old_ann_dt]
    return target_df


def fetch_data(target_df):
    '''
    get data fromm wind
    Args:
        target_df: DataFrame,

    Returns:DataFrame

    '''
    name_df = pd.read_excel(os.path.join(DIR,'indicators_name.xlsx'), sheet_name='financial')
    dfs = []
    for date, row in target_df.iterrows():
        row = row.dropna()
        if len(row)>0:
            row = row.apply(lambda x: x.strftime(DATE_FORMAT))
            rpdate = row.name.strftime(DATE_FORMAT)
            stkcds = row.index.tolist()
            indicators = name_df['wind_name'].tolist()
            # _df=pd.concat(multi_process(get_data_for_one_stk,((stkcd, rpdate, indicators) for stkcd in stkcds),5,
            #                             multi_paramters=True), axis=1).T
            # _df=pd.concat(multi_thread(get_data_for_one_stk,((stkcd, rpdate, indicators) for stkcd in stkcds)),axis=1).T

            _df = pd.concat([get_data_for_one_stk(stkcd, rpdate, indicators) for stkcd in stkcds], axis=1,sort=True).T
            _df['rpdate'] = rpdate
            _df.set_index('rpdate', append=True, inplace=True)
            _df.index.names = ['stkcd', 'rpdate']
            _df=_df.swaplevel()
            dfs.append(_df)
    df = pd.concat(dfs,sort=True)

    return df

def construct_original_data():
    ann_dt=get_ann_dt_df()
    for i in [10,17,18]:
        ann=ann_dt.iloc[:,:i]
        df=fetch_data(ann)
        df.to_pickle(os.path.join(DIR,'data_2018-08-{}.pkl'.format(i)))

def update(date=None):
    if date is None:
        date=get_today(DATE_FORMAT)

    target_df=get_target_df(date) #fixme:
    if target_df.notnull().sum().sum()>0:
        data=fetch_data(target_df)
        latest_data=get_latest(date,'data')
        df=pd.concat([data,latest_data],sort=True).drop_duplicates()
    else:
        df=get_latest(date,'data')

    df.to_pickle(os.path.join(DIR,f'data_{date}.pkl'))
    df.to_csv(os.path.join(DIR,f'data_{date}.csv'))

def debug():
    dates=pd.date_range('2018-08-17','2018-09-11')
    dates=[d.strftime('%Y-%m-%d') for d in dates]
    for date in dates:
        try:
            print(f'Updating data for ------>{date}')
            update(date)
        except:
            pass

'''
w.wss("600000.SH,600004.SH,600006.SH,600007.SH,600008.SH,600009.SH", "west_netprofit_FY1,west_netprofit_FY2,west_netprofit_FY3,west_sales_FY1,west_sales_FY2,west_sales_FY3,west_avgbps_FY1,west_avgbps_FY2,west_avgbps_FY3,west_avgshares","unit=1;tradeDate=20180918;year=2018;westPeriod=180")


'''


if __name__ == '__main__':
    debug()




# if __name__ == '__main__':
#     update()




#TODO: logger and multiprocessing


#TODO: change the indictor names
#TODO: only update in trading date

