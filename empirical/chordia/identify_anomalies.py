# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-26  12:18
# NAME:FT_hp-identify_anomalies.py
from config import DIR_TMP
from data.dataApi import read_local
from empirical.bootstrap import pricing_assets
from empirical.config_ep import DIR_CHORDIA, DIR_DM, DIR_DM_INDICATOR, \
    DIR_DM_NORMALIZED
import os
import pandas as pd
from empirical.get_basedata import get_raw_factors, get_benchmark, BENCHS
import numpy as np
import statsmodels.formula.api as sm
from empirical.utils import align_index
from tools import multi_process

START='2005'
END='2017'

#--------------------hedged portfolio--------------------------
def get_all_return_factors():
    path=os.path.join(DIR_TMP,'dtesatfdaf.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        directory = os.path.join(DIR_DM, 'port_ret', 'eq')
        fns = os.listdir(directory)
        arg_generator = (os.path.join(directory, fn) for fn in fns)
        # ss=multi_process(_get_tb, arg_generator)
        # ss=multi_thread(_get_tb,arg_generator)
        _get_tb=lambda path:pd.read_pickle(path)['tb']

        ss = [_get_tb(arg) for arg in arg_generator]
        raw_factors = pd.concat(ss, axis=1, keys=[fn[:-4] for fn in fns])
        # trick: delete those months with too small sample
        raw_factors=raw_factors.loc[START:END]
        raw_factors = raw_factors.dropna(axis=1, thresh=int(raw_factors.shape[0] * 0.8))
        raw_factors=raw_factors.fillna(0)

        raw_factors.to_pickle(path)
        return raw_factors

def pricing_all_factors(model,name):
    assets=get_all_return_factors()
    model,assets=align_index(model,assets)
    result=pricing_assets(model,assets)
    s=result['alpha_t'].sort_values()
    s.name=name
    return s


def get_alpha_t_for_all_bm():
    for bench in BENCHS:
        print(bench)
        bendf=get_benchmark(bench).dropna()#.ffill()
        s=pricing_all_factors(bendf,bench)
        s.to_pickle(os.path.join(DIR_CHORDIA,f't_{bench}.pkl'))


# get_alpha_t_for_all_bm()

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

CONTROL=['amihud','bp','cumRet','log_size','turnover','ivol']

def _get_other_data():
    #controlling variables,copy from E:\haitong_new\raw_data

    ss = []
    for ct in CONTROL:
        s = pd.read_pickle(
            r'G:\FT_Users\HTZhang\haitong\standardized\{}.pkl'.format(
                ct)).stack()
        s.name = ct
        ss.append(s)

    other = pd.concat(ss, axis=1, join='inner').groupby('month_end').shift(1)
    other = other.dropna()

    other=other.groupby('month_end').filter(lambda s:len(s)>200)#trick: at least 200 stocks in each month
    ret=read_local('trading_m')['ret_m'].swaplevel()
    other=pd.concat([other,ret],axis=1).dropna()
    return other

# other=_get_other_data()
# other.to_pickle(os.path.join(DIR_TMP,'other.pkl'))
other=pd.read_pickle(os.path.join(DIR_TMP,'other.pkl'))


_convert_name=lambda s:s[1:].replace('_','').replace('-','')

def _read_pkl(name):
    s=pd.read_pickle(os.path.join(DIR_DM_NORMALIZED,name+'.pkl')).stack()
    s.name=_convert_name(name)
    print(name)
    return s


def _get_fm_tvalue(name):
    signal = pd.read_pickle(
        os.path.join(DIR_DM_NORMALIZED, name + '.pkl')).stack()
    newname = _convert_name(name)
    signal.name = newname
    comb = pd.concat([other, signal], axis=1)
    comb = comb.dropna()
    if len(comb) > 0:
        formula = f'ret_m ~ {newname} + amihud + bp + cumRet + log_size + turnover + ivol'
        results = famaMacBeth(formula, 'month_end', comb)
        results.to_pickle(os.path.join(DIR_CHORDIA,'fm',name+'.pkl'))
    print(name)

def get_fm_tvalues():
    fns=os.listdir(DIR_DM_NORMALIZED)
    names=[fn[:-4] for fn  in fns]
    print(len(names))
    checked=[fn[:-4] for fn in os.listdir(os.path.join(DIR_CHORDIA,'fm'))]
    names=[n for n in names if n not in checked]
    print(len(names))
    multi_process(_get_fm_tvalue,names,20)

def _get_t(fn):
    return pd.read_pickle(os.path.join(DIR_CHORDIA,'fm',fn)).at[_convert_name(fn[:-4]),'tvalue']

def get_tvalues_series():
    # _get_tvalue=lambda x:pd.read_pickle(os.path.join(DIR_CHORDIA,'fm',x)).at[_convert_name(x[:-4]),'tvalue']
    fns=os.listdir(os.path.join(DIR_CHORDIA,'fm'))
    s=pd.Series(multi_process(_get_t,fns),index=[fn[:-4] for fn in fns]).sort_values()
    # s=pd.Series([_get_tvalue(fn) for fn in fns],index=[fn[:-4] for fn in fns]).sort_values()
    s.to_csv(os.path.join(DIR_CHORDIA,'fm_t.csv'))


def analyze():
    alpha_t = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f't_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)

    # alpha_t = pd.read_csv(os.path.join(DIR_CHORDIA, 'alpha_t.csv'), index_col=0)

    fm_t=pd.read_csv(os.path.join(DIR_CHORDIA,'fm_t.csv'),index_col=0,header=None)
    fm_t=fm_t.iloc[:,0]
    fm_t=fm_t.replace([-np.inf,np.inf],np.nan)

    CRITIC=3

    inds1=alpha_t[alpha_t>CRITIC].dropna().index
    inds2=alpha_t[alpha_t<-CRITIC].dropna().index
    inds3=fm_t[fm_t>CRITIC].index
    inds4=fm_t[fm_t<-CRITIC].index

    indicators=inds1+inds2
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    df.cumsum().plot().get_figure().show()

    cr=df.corr().stack().sort_values()

    df2=pd.concat([_get_s(ind) for ind in inds4],axis=1,keys=inds4)
    df2.cumsum().plot().get_figure().show()



#-----------------------------aggregate anomalies------------------------------------------

#=================method 0: select manually=================================
def get_prominent_anomalies():
    alpha_t = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f't_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)

    fm_t = pd.read_csv(os.path.join(DIR_CHORDIA, 'fm_t.csv'), index_col=0,
                       header=None)
    fm_t = fm_t.iloc[:, 0]
    fm_t = fm_t.replace([-np.inf, np.inf], np.nan)

    CRITIC = 3

    inds1 = alpha_t[alpha_t > CRITIC].dropna().index.tolist()
    inds2 = alpha_t[alpha_t < -CRITIC].dropna().index.tolist()
    inds3 = fm_t[fm_t > CRITIC].index.tolist()
    inds4 = fm_t[fm_t < -CRITIC].index.tolist()

    indicators=inds1+inds2
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    # df.cumsum().plot().get_figure().show()

    cr=df.corr().stack().sort_values()

    test_indicators=abs(cr).sort_values().index[0]

    myfactors=df[list(test_indicators)]
    return myfactors

def get_alpha_t_of_mymodel():
    ff3=get_benchmark('ff3M')
    myfactors=get_prominent_anomalies()
    mymodel=pd.concat([ff3,myfactors],axis=1).dropna()

    results=pricing_all_factors(mymodel,'mymodel')
    results.to_pickle(os.path.join(DIR_CHORDIA,'t_mymodel.pkl'))



#=================method 1: out-of-sample FM regression------------------------------------



#==================method2: PCA=============================
def get_prominent_anomalies1():
    alpha_t = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f't_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)

    fm_t = pd.read_csv(os.path.join(DIR_CHORDIA, 'fm_t.csv'), index_col=0,
                       header=None)
    fm_t = fm_t.iloc[:, 0]
    fm_t = fm_t.replace([-np.inf, np.inf], np.nan)

    CRITIC = 3

    inds1 = alpha_t[alpha_t > CRITIC].dropna().index.tolist()
    inds2 = alpha_t[alpha_t < -CRITIC].dropna().index.tolist()
    inds3 = fm_t[fm_t > CRITIC].index.tolist()
    inds4 = fm_t[fm_t < -CRITIC].index.tolist()

    indicators=inds1+inds2
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    return df



def _get_pca_model(n=NUM_FACTOR):
    pca_factors=_get_pca_factors(n=n - 1)
    rpM=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata','rpM.pkl'))
    pca_model=pd.concat([rpM,pca_factors],axis=1).dropna()
    return pca_model





#=================method3: cluster=========================

#==================method4: PLS============================


def main():
    get_alpha_t_for_all_bm()
    get_fm_tvalues()
    get_tvalues_series()

#TODO: we should check absolute tvalue, since these signals are generated randomly, being negative or positive does not make any sense.

'''
1. 有些指标样本太少
2. fm 中的指标



'''
