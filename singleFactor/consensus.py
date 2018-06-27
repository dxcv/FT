# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-05  14:19
# NAME:FT-con.py
from config import SINGLE_D_INDICATOR
from data.dataApi import read_raw, read_local
import pandas as pd
import os

def _save(s):
    new_name='C__'+s.name
    df=s.to_frame()
    df.columns=[new_name]
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,new_name+'.pkl'))


def _get_monthly_data():
    con=read_raw('equity_consensus_forecast')

    con['trd_dt']=pd.to_datetime(con['trd_dt'].map(str))
    con['benchmark_yr']=pd.to_datetime(con['benchmark_yr'].map(lambda x:str(x).split('.')[0]))
    con['rating_dt']=pd.to_datetime(con['rating_dt'].map(lambda x:str(x).split('.')[0]))

    trading=read_local('equity_selected_trading_data')
    comb=pd.merge(con,trading.reset_index()[['trd_dt','stkcd','close']],on=['stkcd','trd_dt'],how='left')#使用未复权的收盘价
    # comb=comb[-int(comb.shape[0]/100):] #debug:

    monthly=comb.groupby('stkcd').resample('M',on='trd_dt').last()
    monthly.index.names=['stkcd','month_end']
    return monthly

def apply_g(df,col,period=1):
    '''growth'''
    return df.groupby('stkcd')[col].pct_change(periods=period)

def apply_chg(df,col,period=1):
    '''absolute change'''
    return df.groupby('stkcd')[col].apply(lambda x:x-x.shift(period))

def cal():
    monthly=_get_monthly_data()
    old_columns=monthly.columns.tolist()
    indicators=['est_net_profit','est_oper_revenue','est_bookvalue']
    lengths=['FTTM','FT24M']
    scale='est_baseshare'

    for indicator in indicators:
        for length in lengths:
            # monthly['{}_{}_{}'.format(indicator,length,'level')]=monthly['{}_{}'.format(indicator,length)]
            per='{}_{}_{}'.format(indicator,length,'perShare')
            monthly[per]=monthly['{}_{}'.format(indicator,length)]/monthly['{}_{}'.format(scale,length)]
            for period in [1,3,6]:
                monthly['{}_{}_{}'.format(indicator,length,'perShare_g_{}m'.format(period))]=apply_g(monthly,per,period=period)
                monthly['{}_{}_{}'.format(indicator,length,'perShare_chg_{}m'.format(period))]=apply_chg(monthly,per,period=period)
            print(indicator,length)


    for day in [30,90,180]:
        monthly['est_price_{}_relative'.format(day)]=monthly['est_price_{}'.format(day)]/monthly['close']
        for period in [1,3,6]:
            monthly['{}_{}_{}_{}'.format('est_price',day,'g',period)]=apply_g(monthly,'est_price_{}'.format(day),period=period)
            monthly['{}_{}_{}_{}'.format('est_price',day,'chg',period)]=apply_chg(monthly,'est_price_{}'.format(day),period=period)

    del monthly['stkcd']
    monthly=monthly.reset_index().set_index(['stkcd','trd_dt'])

    for col in monthly.columns:
        if col not in old_columns+['month_end']:
            _save(monthly[col])

if __name__ == '__main__':
    cal()
