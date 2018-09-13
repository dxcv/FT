# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-30  16:59
# NAME:FT_hp-identify_anomalies1.py
import shutil
from collections import OrderedDict
import statsmodels.formula.api as sm

from config import DIR_TMP
from data.dataApi import read_local, get_filtered_ret
from empirical.bootstrap import pricing_assets
from empirical.config_ep import DIR_DM, DIR_CHORDIA, DIR_DM_NORMALIZED, \
    PERIOD_THRESH, DIR_BASEDATA, DIR_YAN
import os
import pandas as pd
from empirical.get_basedata import BENCHS, get_benchmark, CONTROL, get_data
from empirical.utils import align_index, get_pca_factors
from empirical.yan.yan_new import get_realized
from tools import multi_process, z_score
import numpy as np
import pickle
import matplotlib.pyplot as plt


START='2005'
END='2017'

#--------------------hedged portfolio--------------------------
def pricing_all_factors(model):
    spread=pd.read_pickle(os.path.join(DIR_DM,'spread.pkl'))
    # assets=get_all_return_factors()
    model,assets=align_index(model,spread)
    result=pricing_assets(model,assets)
    s=result['alpha_t'].sort_values()
    return s

def pricing_all_factors_old(model,name):
    spread=pd.read_pickle(os.path.join(DIR_DM,'spread.pkl'))
    # assets=get_all_return_factors()
    model,assets=align_index(model,spread)
    result=pricing_assets(model,assets)
    s=result['alpha_t'].sort_values()
    s.name=name
    return s

def get_alpha_t_for_all_bm():
    for bname in BENCHS:
        print(bname)
        bench=get_benchmark(bname).dropna()#.ffill()
        if isinstance(bench, pd.Series):
            bench = bench.to_frame()
        s=pricing_all_factors(bench)
        s.to_pickle(os.path.join(DIR_CHORDIA,f'at_{bname}.pkl'))#alpha t value


#--------------------------------FM regression-------------------------------
def famaMacBeth(formula, time_label, df, lags=5):
    res = df.groupby(time_label,sort=True).apply(lambda x: sm.ols(
        formula, data=x).fit())
    p=pd.DataFrame([x.params for x in res],index=res.index)
    means = {}
    for x in p.columns:
        if lags is 0:
            means[x] = sm.ols(formula=x + ' ~ 1',
                              data=p[[x]]).fit(use_t=True)
        else:
            means[x] = sm.ols(formula=x + ' ~ 1',
                              data=p[[x]]).fit(cov_type='HAC',
                                cov_kwds={'maxlags': lags},
                                use_t=True)
    result=pd.DataFrame([
        [means[x].params['Intercept'],means[x].bse['Intercept'],
         means[x].tvalues['Intercept'],means[x].pvalues['Intercept']]
        for x in p.columns
    ],index=p.columns,columns=['coef','stderr','tvalue','pvalue'])
    result['stars']=result['pvalue'].map(lambda x:'***' if x<0.01 else ('**' if x<0.05 else ('*' if x<0.1 else '')))
    return result

# CONTROL=['amihud','bp','cumRet','log_size','turnover','ivol']




def _get_other_data():
    #controlling variables,copy from E:\haitong_new\raw_data

    ss = []
    for ct in CONTROL:
        # s = pd.read_pickle(
        #     r'G:\FT_Users\HTZhang\haitong\standardized\{}.pkl'.format(
        #         ct)).stack()
        s=pd.read_pickle(os.path.join(DIR_BASEDATA,'normalized_controlling',ct+'.pkl'))
        # s.name = ct
        ss.append(s)

    other = pd.concat(ss, axis=1, join='inner').groupby('stkcd').shift(1) #trick: use the indicator of time t-1
    other = other.dropna(how='all')
    other=other.groupby('month_end').filter(lambda s:len(s)>200)#trick: at least 200 stocks in each month
    ret=get_filtered_ret().swaplevel()
    # ret=read_local('trading_m')['ret_m'].swaplevel()
    other=pd.concat([other,ret],axis=1).dropna()
    other=other.fillna(0)
    return other

_convert_name=lambda s:s[1:].replace('_','').replace('-','')


control_map=OrderedDict({
    'capmM':[],
    'ff3M':['log_size','bm'],
    'ffcM':['log_size','bm','mom'],
    'ff5M':['log_size','bm','op','inv'],
    'ff6M':['log_size','bm','op','inv','mom'],
    'hxz4M':['log_size','bm','inv','roe']
})


# other=_get_other_data()
# other.to_pickle(os.path.join(DIR_TMP,'other.pkl'))

other=pd.read_pickle(os.path.join(DIR_TMP,'other.pkl'))



def _get_fmt(name):
    signal = pd.read_pickle(
        os.path.join(DIR_DM_NORMALIZED, name + '.pkl')).shift(1).stack()#trick:use the indicator of time t-1
    newname = _convert_name(name)
    signal.name = newname
    comb = pd.concat([other, signal], axis=1)
    comb = comb.dropna()
    comb=comb.groupby('month_end').filter(lambda df:len(df)>300) #trick: at least 300 stocks in each month
    if len(set(comb.index.get_level_values('month_end')))>PERIOD_THRESH:#trick: at least 60 periods
        ts=[]
        for mn in control_map:
            if len(control_map[mn])>0:
                formula=f'ret_m ~ {newname}' + ' + ' + ' + '.join(control_map[mn])
            else:
                formula=f'ret_m ~ {newname}'

        # formula = f'ret_m ~ {newname} + amihud + bp + cumRet + log_size + turnover + ivol'
            tvalue = famaMacBeth(formula, 'month_end', comb).at[newname,'tvalue']
            ts.append(tvalue)
            # print(mn)
        ts=pd.Series(ts,index=control_map.keys())
        ts.name=name
        ts.to_pickle(os.path.join(DIR_CHORDIA,'fm',name+'.pkl'))
        print(name)

def calculate_fmt():
    names=pickle.load(open(os.path.join(DIR_DM,'playing_indicators.pkl'),'rb'))
    print('total',len(names))
    checked=[fn[:-4] for fn in os.listdir(os.path.join(DIR_CHORDIA,'fm'))]
    names=[n for n in names if n not in checked]#fixme:
    print('remainder',len(names))

    multi_process(_get_fmt, names, 20)

    # for name in names:
    #     _get_fmt(name)

# if __name__ == '__main__':
#     calculate_fmt()
#

def _get_t(fn):
    return pd.read_pickle(os.path.join(DIR_CHORDIA,'fm',fn))

def get_fmt():
    # _get_tvalue=lambda x:pd.read_pickle(os.path.join(DIR_CHORDIA,'fm',x)).at[_convert_name(x[:-4]),'tvalue']
    fns=os.listdir(os.path.join(DIR_CHORDIA,'fm'))
    df=pd.concat(multi_process(_get_t,fns),axis=1).T
    df.to_pickle(os.path.join(DIR_CHORDIA,'fmt.pkl'))

# if __name__ == '__main__':
#     get_fmt()



def get_prominent_indicators(critic=3):
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)
    fmt=pd.read_pickle(os.path.join(DIR_CHORDIA,'fmt.pkl'))

    inds1=at[abs(at)>critic].dropna().index.tolist()
    inds2=fmt[abs(fmt)>critic].dropna().index.tolist()
    inds=[ind for ind in inds1 if ind in inds2]
    # len(inds) #26

    # inds1=at[at>CRITIC].dropna().index.tolist()
    # inds2=at[at<-CRITIC].dropna().index.tolist()
    # inds3=fmt[fmt>CRITIC].dropna().index.tolist()
    # inds4=fmt[fmt<-CRITIC].dropna().index.tolist()

    # _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    # df=pd.concat([_get_s(ind) for ind in inds],axis=1,keys=inds)
    # df.cumsum().plot().get_figure().show()

    # cr=df.corr().stack().sort_values()
    return inds

#-----------------------------aggregate anomalies------------------------------------------

#=================method 0: select manually=================================
def get_prominent_anomalies0():
    alpha_t = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)
    fmt = pd.read_pickle(os.path.join(DIR_CHORDIA, 'fmt.pkl'))

    CRITIC = 3

    inds1 = alpha_t[alpha_t > CRITIC].dropna().index.tolist()
    inds2 = alpha_t[alpha_t < -CRITIC].dropna().index.tolist()
    inds3 = fmt[fmt > CRITIC].index.tolist()
    inds4 = fmt[fmt < -CRITIC].index.tolist()

    indicators=inds1+inds2
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    # df.cumsum().plot().get_figure().show()

    cr=df.corr().stack().sort_values()

    test_indicators=abs(cr).sort_values().index[0]

    myfactors=df[list(test_indicators)]
    return myfactors

def get_at_manuallymodel():
    ff3=get_benchmark('ff3M')
    myfactors=get_prominent_anomalies0()
    manually=pd.concat([ff3,myfactors],axis=1).dropna()

    results=pricing_all_factors(manually,'manually')
    results.to_pickle(os.path.join(DIR_CHORDIA,'at_manually.pkl'))

    #compare
    alpha_t = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)
    at_my=pd.read_pickle(os.path.join(DIR_CHORDIA,'at_mymodel.pkl'))
    (abs(alpha_t)>3).sum()
    (abs(at_my)>3).sum()

#=================method3: cluster=========================

#==================method4: PLS============================

#==================================================================================================
def main():
    get_alpha_t_for_all_bm()
    calculate_fmt()
    get_fmt()

#TODO: we should check absolute tvalue, since these signals are generated randomly, being negative or positive does not make any sense.

'''
1. 有些指标样本太少
2. fm 中的指标

'''

#===========================method5: bootstrap==================================
bench='ff3M'
benchmark, assets = get_data(bench)
simulated=pickle.load(open(os.path.join(DIR_YAN,f'{bench}_100.pkl'),'rb'))
for bench in BENCHS:
    if bench not in ['ffcM']:
        get_realized(bench)
        print(bench)



# bt=simulated['alpha_t']


