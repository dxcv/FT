# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-05  14:19
# NAME:FT-con.py
from data.dataApi import read_local_pkl, read_raw
import pandas as pd
from singleFactor.factors.check import _check

con=read_raw('equity_consensus_forecast')
con=con[-int(con.shape[0]/100):]



# 预测 12 个月净利润  e_net_profit_12m
con['trd_dt']=pd.to_datetime(con['trd_dt'].map(str))
con_m=con.groupby('stkcd').resample('M', on='trd_dt').last()
con_m.index.names=['stkcd','month_end']
con_m=con_m.reset_index(drop=True).set_index(['stkcd','trd_dt'])
_check(con_m[['est_net_profit_FTTM']],'e_net_profit_12m')


# 预测12个月eps
con_m['e_eps']=con_m['est_net_profit_FTTM']/con_m['est_baseshare_FTTM']
_check(con_m[['e_eps']],'e_eps_12m')

#预测 12 个月的 eps-3 个月之前的预测值
con_m['e_eps_12m_chg_3m']=con_m['e_eps'].groupby('stkcd').pct_change(3)
_check(con_m[['e_eps_12m_chg_3m']],'e_eps_12m_chg_3m')

#预测 12 个月的 eps-6 个月之前的预测值
con_m['e_eps_12m_chg_6m']=con_m['e_eps'].groupby('stkcd').pct_change(6)
_check(con_m[['e_eps_12m_chg_6m']],'e_eps_12m_chg_6m')








