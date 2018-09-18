# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-16  16:02
# NAME:FT_hp-1 generate_indicators_gta.py
from empirical.config_ep import DIR_DM_GTA
import pandas as pd
import os
import numpy as np
from tools import multi_process

START='2000'


'''
financial is prepared by G:\\backup\\code\\assetPricing2\\used_outside_project\\get_all_financial_indicators.py

'''
financial=pd.read_pickle(os.path.join(DIR_DM_GTA,'financial.pkl')).swaplevel().sort_index()
valid=[cd for cd in set(financial.index.get_level_values('stkcd')) if cd[0] in (['0','3','6'])]
financial=financial.loc[(slice(None),valid),:]

financial=financial.replace(0.0,np.nan)#trick: replace 0 with np.nan

with open(os.path.join(DIR_DM_GTA,'base_variables.txt')) as f:
    lines=f.read().split('\n')
    base_variables=[l.split('\t')[0] for l in lines]

other_variables=[col for col in financial.columns if col not in base_variables]

# basedf=financial[base_variables]
# otherdf=financial[other_variables]


#TODO：mktcap
save_s=lambda s,name:s.to_pickle(os.path.join(DIR_DM_GTA,'indicators_yearly',name+'.pkl'))

def ratio(other,base):
    name=f'ratio-{other}-{base}'
    s=financial[other]/financial[base]
    save_s(s,name)

def ratio_chg(other,base):
    name=f'ratio_chg-{other}-{base}'
    _df = (financial[other] / financial[base]).unstack('stkcd')
    s = (_df - _df.shift(1)).stack()
    save_s(s,name)

def ratio_growth(other,base):
    name=f'ratio_growth-{other}-{base}'
    s = (financial[other] / financial[base]).unstack(
        'stkcd').pct_change().stack()
    save_s(s,name)

def ratio_x_chg_over_lag_y(other,base):
    name=f'ratio_x_chg_over_lag_y-{other}-{base}'
    s = (financial[other] - financial[other].groupby('stkcd').shift(1)) / \
         financial[base].groupby('stkcd').shift(1)
    save_s(s,name)

def ratio_growth_dif(other,base):
    name=f'ratio_growth_dif-{other}-{base}'
    s = financial[other].groupby('stkcd').pct_change() - financial[base].groupby(
        'stkcd').pct_change()
    save_s(s,name)

def x_growth(other):
    name=f'x_growth-{other}'
    s = financial[other].groupby('stkcd').pct_change()
    save_s(s,name)

def one_other_base(other,base):
    ratio(other,base)
    ratio_chg(other,base)
    ratio_growth(other,base)
    ratio_x_chg_over_lag_y(other,base)
    ratio_growth_dif(other,base)
    x_growth(other)

def generate_all():
    args_list=((other,base) for other in other_variables for base in base_variables)
    multi_process(one_other_base,args_list,20,multi_parameters=True)

def _yearly2monthly(fn):
    df=pd.read_pickle(os.path.join(DIR_DM_GTA,'indicators_yearly',fn))
    df=df.unstack('stkcd')
    dr=pd.date_range(start='1990',end='2019',freq='M')
    df=df.reindex(dr).shift(6).ffill(limit=11)
    df=df.dropna(how='all')
    if df.shape[0]>0:
        df.to_pickle(os.path.join(DIR_DM_GTA,'indicators_monthly',fn))

# monthly=pd.read_pickle(r'G:\FT_Users\HTZhang\empirical\data_mining\indicator\1-x_chg-cap_rsrv\monthly.pkl')

def yearly2monthly():
    fns=os.listdir(os.path.join(DIR_DM_GTA,'indicators_yearly'))
    multi_process(_yearly2monthly,fns,30)


def main():
    generate_all()
    yearly2monthly()


if __name__ == '__main__':
    main()


