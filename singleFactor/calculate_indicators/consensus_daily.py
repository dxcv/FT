# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-12  10:05
# NAME:FT_hp-consensus_daily.py
from config import SINGLE_D_INDICATOR
from data.dataApi import read_raw, read_local
import pandas as pd
import os


FORWARD_LIMIT=20
SMOOTH_PERIOD=30



def save_indicator(df,name):
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))


def cal():
    con = read_raw('equity_consensus_forecast')
    # con = read_raw('equity_consensus_forecast')

    con['trd_dt'] = pd.to_datetime(con['trd_dt'])
    # con['benchmark_yr'] = pd.to_datetime(
    #     con['benchmark_yr'].map(lambda x: str(x).split('.')[0]))
    con['benchmark_yr']=pd.to_datetime(con['benchmark_yr'])
    # con['rating_dt'] = pd.to_datetime(
    #     con['rating_dt'].map(lambda x: str(x).split('.')[0]))

    con['rating_dt']=pd.to_datetime(con['rating_dt'])

    trading = read_local('equity_selected_trading_data')
    comb = pd.merge(trading.reset_index()[['trd_dt', 'stkcd', 'close']],con,
                    on=['stkcd', 'trd_dt'], how='left')  # 使用未复权的收盘价


    types = ['est_net_profit', 'est_oper_revenue', 'est_bookvalue']
    lengths = ['FTTM', 'FT24M']
    days=[20,60,180]

    for l in lengths:
        for type in types:
            numerator='_'.join([type,l])
            for denominator in ['est_baseshare_{}'.format(l),'close']:
                indName='{}_to_{}'.format(numerator,denominator)
                comb[indName]=comb[numerator]/comb[denominator]
                d=pd.pivot_table(comb, values=indName, index='trd_dt', columns='stkcd')
                d=d.ffill(limit=FORWARD_LIMIT)
                # d=d.rolling(SMOOTH_PERIOD).mean()#trick： 平滑，否则换手率太高
                #trick: 如果月频的测出来有效，我们可以把日度的数据变为月度的。每月调仓一次就好了
                for day in days:
                    nameG='{}_g_{}'.format(indName,day)
                    g=d.pct_change(periods=day)
                    save_indicator(g,'C__{}'.format(nameG))
                    nameC='{}_chg_{}'.format(indName,day)
                    chg=d-d.shift(day)
                    save_indicator(chg,'C__{}'.format(nameC))
                    print(numerator,denominator,day)

if __name__ == '__main__':
    cal()



