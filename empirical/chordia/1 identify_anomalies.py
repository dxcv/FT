# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-30  16:59
# NAME:FT_hp-1 identify_anomalies.py
import shutil
from collections import OrderedDict

from config import DIR_TMP
from data.dataApi import read_local, get_filtered_ret
from empirical.bootstrap import pricing_assets
from empirical.config_ep import DIR_DM, DIR_CHORDIA, DIR_DM_NORMALIZED, \
    PERIOD_THRESH, DIR_BASEDATA
import os
import statsmodels.api as sm
import pandas as pd
from empirical.get_basedata import BENCHS, get_benchmark, CONTROL
from empirical.utils import align_index, get_pca_factors
from tools import multi_process, z_score
import numpy as np
import pickle
import matplotlib.pyplot as plt


START='2005'
END='2017'

#--------------------hedged portfolio--------------------------
def pricing_all_factors(model,name):
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
        s=pricing_all_factors(bench,bname)
        s.to_pickle(os.path.join(DIR_CHORDIA,f'at_{bname}.pkl'))#alpha t value


#--------------------------------FM regression-------------------------------
def famaMacBeth(formula, time_label, df, lags=5):
    import statsmodels.formula.api as sm
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

# other.to_pickle(os.path.join(DIR_TMP,'other.pkl'))
# other=pd.read_pickle(os.path.join(DIR_TMP,'other.pkl'))
_convert_name=lambda s:s[1:].replace('_','').replace('-','')


control_map=OrderedDict({
    'capmM':[],
    'ff3M':['log_size','bm'],
    'ffcM':['log_size','bm','mom'],
    'ff5M':['log_size','bm','op','inv'],
    'ff6M':['log_size','bm','op','inv','mom'],
    'hxz4M':['log_size','bm','inv','roe']
})


def _get_fmt(name):
    other = _get_other_data()

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
        ts=pd.Series(ts,index=control_map.keys())
        ts.name=name
        ts.to_pickle(os.path.join(DIR_CHORDIA,'fm',name+'.pkl'))
        print(name)


def calculate_fmt():
    names=pickle.load(open(os.path.join(DIR_DM,'playing_indicators.pkl'),'rb'))
    checked=[fn[:-4] for fn in os.listdir(os.path.join(DIR_CHORDIA,'fm'))]
    names=[n for n in names if n not in checked]
    print(len(names))
    # multi_process(_get_fmt, names, 15)
    for name in names:
        _get_fmt(name)

if __name__ == '__main__':
    calculate_fmt()



def _get_t(fn):
    return pd.read_pickle(os.path.join(DIR_CHORDIA,'fm',fn)).at[_convert_name(fn[:-4]),'tvalue']

def get_fmt_series():
    # _get_tvalue=lambda x:pd.read_pickle(os.path.join(DIR_CHORDIA,'fm',x)).at[_convert_name(x[:-4]),'tvalue']
    fns=os.listdir(os.path.join(DIR_CHORDIA,'fm'))
    s=pd.Series(multi_process(_get_t,fns),index=[fn[:-4] for fn in fns]).sort_values()
    # s=pd.Series([_get_tvalue(fn) for fn in fns],index=[fn[:-4] for fn in fns]).sort_values()
    s.to_pickle(os.path.join(DIR_CHORDIA,'fmt.pkl'))


# if __name__ == '__main__':
#     get_fmt_series()

def analyze():
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)
    fmt=pd.read_pickle(os.path.join(DIR_CHORDIA,'fmt.pkl'))

    CRITIC=3

    inds1=at[at>CRITIC].dropna().index.tolist()
    inds2=at[at<-CRITIC].dropna().index.tolist()
    inds3=fmt[fmt>CRITIC].index.tolist()
    inds4=fmt[fmt<-CRITIC].index.tolist()

    indicators=inds1+inds2
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    df.cumsum().plot().get_figure().show()

    cr=df.corr().stack().sort_values()

    df2=pd.concat([_get_s(ind) for ind in inds3[:20]],axis=1,keys=inds3[:20])
    df2.cumsum().plot().get_figure().show()



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

#=================method 1: out-of-sample FM regression------------------------------------
def get_prominent_anomalies1():
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)

    fmt = pd.read_pickle(os.path.join(DIR_CHORDIA, 'fmt.pkl'))

    CRITIC = 3

    inds1 = at[at > CRITIC].dropna().index.tolist()
    inds2 = at[at < -CRITIC].dropna().index.tolist()
    return inds1+inds2

def fm_predict(indicator, smooth_period=1):
    '''

    Args:
        indicator:DataFrame with multiIndex, does not include the 'const' column
        smooth_period:

    Returns:

    '''
    # ret = read_local('trading_m')['ret_m'].swaplevel()
    ret=get_filtered_ret().swaplevel()
    ind_inter = indicator.index.intersection(ret.index)
    indicator = indicator.reindex(ind_inter)
    ret = ret.reindex(ind_inter)

    indicator_1m = indicator.groupby('stkcd').shift(
        1)  # trick: use the indicator of time t-1
    comb = pd.concat([indicator_1m, ret], axis=1).dropna()

    comb = sm.add_constant(comb)
    inde = [col for col in comb.columns if col != 'ret_m']
    rs = comb.groupby('month_end').apply(
        lambda df: sm.OLS(df['ret_m'], df[inde]).fit())

    params = pd.concat([r.params for r in rs], axis=1, keys=rs.index).T

    params = params.rolling(smooth_period).mean()
    indicator = indicator.loc[(params.index, slice(None)), :]
    indicator = sm.add_constant(indicator)

    ss = []
    for month in params.index:
        sub = indicator.loc[(month, slice(None)), :]
        pred = np.array(sub) @ np.array(params.loc[month])
        s = pd.Series(pred, index=sub.index.get_level_values('stkcd'))
        s.name = month
        ss.append(s)
        print(month)

    predicted = pd.concat(ss, axis=1, sort=True).T.shift(1)  # trick: use the parameters and value of time t to predicted return in time t+1
    predicted = predicted.stack()
    predicted.name = 'predicted'
    return predicted

def tmb_with_fm_predicted(predicted):
    # ret = read_local('trading_m')['ret_m'].swaplevel()
    ret=get_filtered_ret().swaplevel()
    comb = pd.concat([predicted, ret], axis=1).dropna()
    comb.index.names = ['month_end', 'stkcd']

    comb['g'] = comb.groupby('month_end', group_keys=False).apply(
        lambda df: pd.qcut(df['predicted'].rank(method='first'), 10,
                           labels=['g{}'.format(i) for i in range(1, 11)],
                           duplicates='raise'))

    port_ret_eq = comb.groupby(['month_end', 'g'])['ret_m'].mean().unstack(level=1)
    port_ret_eq.columns = port_ret_eq.columns.astype(str)

    tmb = port_ret_eq['g10'] - port_ret_eq['g1']
    return tmb

def test_n(number):
    inds = get_prominent_anomalies1()
    selected=inds[:number]
    indicator=pd.concat([pd.read_pickle(os.path.join(DIR_DM_NORMALIZED, sl + '.pkl')).stack() for sl in selected], axis=1, keys=selected)
    indicator=indicator.dropna(thresh=int(indicator.shape[1]*0.6))#trick: dropna and fillna with mean values
    indicator=indicator.groupby('month_end').apply(lambda df:df.fillna(df.mean()))

    predicted=fm_predict(indicator)
    tmb=tmb_with_fm_predicted(predicted)
    ff3=get_benchmark('ff3M')
    fmmodel=pd.concat([tmb, ff3], axis=1)
    results = pricing_all_factors(fmmodel, f'fmmodel{number}')
    results.name=number
    return results


def observe_n():
    numbers=[1,3,5,10,15,20,25]
    df=pd.concat(multi_process(test_n,numbers,n=7),axis=1,sort=True)
    df.to_pickle(os.path.join(DIR_CHORDIA,'observe_fm_n.pkl'))


def debug_fm_predict(indicator, smooth_period=1):
    '''

    Args:
        indicator:DataFrame with multiIndex, does not include the 'const' column
        smooth_period:

    Returns:

    '''
    # ret = read_local('trading_m')['ret_m'].swaplevel()
    ret=get_filtered_ret().swaplevel()
    ind_inter = indicator.index.intersection(ret.index)
    indicator = indicator.reindex(ind_inter)
    ret = ret.reindex(ind_inter)

    indicator_1m = indicator.groupby('stkcd').shift(
        1)  # trick: use the indicator of time t-1
    comb = pd.concat([indicator_1m, ret], axis=1).dropna()

    comb = sm.add_constant(comb)
    inde = [col for col in comb.columns if col != 'ret_m']
    rs = comb.groupby('month_end').apply(
        lambda df: sm.OLS(df['ret_m'], df[inde]).fit())

    params = pd.concat([r.params for r in rs], axis=1, keys=rs.index).T

    params = params.rolling(smooth_period).mean()
    indicator = indicator.loc[(params.index, slice(None)), :]
    indicator = sm.add_constant(indicator)

    ss = []
    for month in params.index:
        sub = indicator.loc[(month, slice(None)), :]
        pred = np.array(sub) @ np.array(params.loc[month])
        s = pd.Series(pred, index=sub.index.get_level_values('stkcd'))
        s.name = month
        ss.append(s)
        print(month)

    predicted = pd.concat(ss, axis=1, sort=True).T.shift(1)  # trick: use the parameters and value of time t to predicted return in time t+1
    predicted = predicted.stack()
    predicted.name = 'predicted'



    # ret = read_local('trading_m')['ret_m'].swaplevel()
    ret=get_filtered_ret().swaplevel()
    comb = pd.concat([predicted, ret], axis=1).dropna()
    comb.index.names = ['month_end', 'stkcd']

    comb['g'] = comb.groupby('month_end', group_keys=False).apply(
        lambda df: pd.qcut(df['predicted'].rank(method='first'), 10,
                           labels=['g{}'.format(i) for i in range(1, 11)],
                           duplicates='raise'))

    port_ret_eq = comb.groupby(['month_end', 'g'])['ret_m'].mean().unstack(level=1)
    port_ret_eq.columns = port_ret_eq.columns.astype(str)

    tmb = port_ret_eq['g10'] - port_ret_eq['g1']
    return tmb

def debug():
    ind='2-pct_chg_dif-cash_recp_return_invest-net_profit_excl_min_int_inc'
    indicator = pd.read_pickle(
        os.path.join(DIR_DM_NORMALIZED, ind + '.pkl')).stack()
    indicator.name = ind
    indicator = indicator.to_frame()
    indicator = indicator.dropna()

    predicted = debug_fm_predict(indicator,smooth_period=1)
    predicted.index.names = ['month_end', 'stkcd']



def forecast_combination(smooth=60):
    inds = get_prominent_anomalies1()
    ss=[]
    for ind in inds[::-1]:
        indicator = pd.read_pickle(
            os.path.join(DIR_DM_NORMALIZED, ind + '.pkl')).stack()
        indicator.name = ind
        indicator = indicator.to_frame()
        indicator = indicator.dropna()
        predicted=fm_predict(indicator, smooth)
        predicted.index.names=['month_end','stkcd']
        predicted=predicted.groupby('month_end').apply(z_score)
        ss.append(predicted)

    tmbs=[]
    for n in [1,3,5,10,15,25]:
        p=pd.concat(ss[:n],axis=1).sum(axis=1)
        p.name='predicted'
        tmb=tmb_with_fm_predicted(p)
        tmb.name=n
        tmbs.append(tmb)
    df=pd.concat(tmbs,axis=1)
    df.cumsum().plot()
    plt.savefig(os.path.join(DIR_CHORDIA, f'combination_forecast_{smooth}.png'))

# if __name__ == '__main__':
#     hs=[1,5,10,30,60]
#     multi_process(forecast_combination,hs,n=5)
#

# if __name__ == '__main__':
#     observe_n()




def _tmp_func(ind):
    indicator = pd.read_pickle(
        os.path.join(DIR_DM_NORMALIZED, ind + '.pkl')).stack()
    indicator.name = ind
    indicator = indicator.to_frame()
    indicator = indicator.dropna()
    ss = []
    for i in [1, 5, 10, 60]:
        predicted=fm_predict(indicator,i)
        s=tmb_with_fm_predicted(predicted)
        # s = fm_predict(indicator, i)
        s.name = i
        ss.append(s)

    df = pd.concat(ss, axis=1)

    df.cumsum().plot()
    plt.savefig(os.path.join(DIR_CHORDIA, f'{ind}.png'))
    plt.close()
    print(ind)

    # for number in [1,5]:
    #     selected=inds[:number]
    #     indicator=pd.concat([pd.read_pickle(os.path.join(DIR_DM_NORMALIZED, sl + '.pkl')).stack() for sl in selected], axis=1, keys=selected)
    #     indicator=indicator.dropna(thresh=int(indicator.shape[1]*0.6))#trick: dropna and fillna with mean values
    #     indicator=indicator.groupby('month_end').apply(lambda df:df.fillna(df.mean()))
    #
    #
    #     ss=[]
    #     for i in [1,5,10,60]:
    #         s=fm_predict(indicator,i)
    #         s.name=i
    #         ss.append(s)
    #
    #     df=pd.concat(ss,axis=1)
    #
    #     df.cumsum().plot()
    #     plt.savefig(os.path.join(DIR_CHORDIA,f'{number}.png'))
    #     plt.close()
    #     print(number)

def run_tmp_func():
    inds = get_prominent_anomalies1()
    multi_process(_tmp_func, inds, 25)



#==================method2: PCA=============================
def get_prominent_anomalies2():
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,sort=True)

    fmt = pd.read_pickle(os.path.join(DIR_CHORDIA, 'fmt.pkl'))

    CRITIC = 3

    inds1 = at[at > CRITIC].dropna().index.tolist()
    inds2 = at[at < -CRITIC].dropna().index.tolist()
    inds3 = fmt[fmt > CRITIC].index.tolist()
    inds4 = fmt[fmt < -CRITIC].index.tolist()

    indicators=inds1+inds2
    _get_s=lambda x:pd.read_pickle(os.path.join(DIR_DM,'port_ret','eq',x+'.pkl'))['tb']

    df=pd.concat([_get_s(ind) for ind in indicators],axis=1,keys=indicators)
    return df

# test=get_prominent_anomalies2()
# test.cumsum().plot().get_figure().show()
# test.mean()
#


def get_at_pca():
    df=get_prominent_anomalies2()
    pca=get_pca_factors(df,2)
    ff3=get_benchmark('ff3M')
    pca_model = pd.concat([ff3, pca], axis=1).dropna()


    results = pricing_all_factors(pca_model, 'pca')
    results.to_pickle(os.path.join(DIR_CHORDIA, 'at_pca.pkl'))

    (abs(results>2)).sum()
    (abs(results>3)).sum()

#=================method3: cluster=========================

#==================method4: PLS============================

#===================method5: forecast combination======================









#==================================================================================================
def main():
    get_alpha_t_for_all_bm()
    calculate_fmt()
    get_fmt_series()

#TODO: we should check absolute tvalue, since these signals are generated randomly, being negative or positive does not make any sense.

'''
1. 有些指标样本太少
2. fm 中的指标



'''
