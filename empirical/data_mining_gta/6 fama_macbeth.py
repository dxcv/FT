# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-18  20:26
# NAME:FT_hp-6 fama_macbeth.py


from collections import OrderedDict
import statsmodels.formula.api as sm

# from data.dataApi import get_filtered_ret
# from empirical.config_ep import DIR_DM, DIR_CHORDIA, DIR_DM_NORMALIZED, \
#     PERIOD_THRESH, DIR_BASEDATA
import os
import pandas as pd
from empirical.config_ep import DIR_DM_GTA, DIR_BASEDATA, CROSS_LEAST, \
    PERIOD_THRESH
from empirical.data_mining_gta.dm_api import get_playing_indicators
from tools import multi_process


DIR_ANALYSE= os.path.join(DIR_DM_GTA, 'analyse')


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

CONTROL=['log_size','bm','mom','op','inv','roe']

def _get_other_data():
    #controlling variables,copy from E:\haitong_new\raw_data
    path=os.path.join(DIR_ANALYSE,'other.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        ss = []
        for ct in CONTROL:
            s=pd.read_pickle(os.path.join(DIR_DM_GTA,ct+'.pkl')).shift(1).stack()#trick: use the indicator of time t-1
            s.name = ct
            ss.append(s)

        other = pd.concat(ss, axis=1, join='inner')
        other.index.names=['month_end','stkcd']
        other = other.dropna(how='all')
        other=other.groupby('month_end').filter(lambda s:len(s)>CROSS_LEAST)#trick: at least 300 stocks in each month
        ret=pd.read_pickle(os.path.join(DIR_DM_GTA, 'fdmt_m.pkl'))['ret_m']
        other=pd.concat([other,ret],axis=1).dropna()
        other=other.fillna(0)
        other.to_pickle(path)
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

other=_get_other_data()

def _get_fmt(name):
    '''
    get tvalue of the alpha in fama macbeth regression
    Args:
        name:str,indicator name

    Returns:

    '''
    signal = pd.read_pickle(
        os.path.join(DIR_DM_GTA,'normalized', name + '.pkl')).shift(1).stack()#trick:use the indicator of time t-1
    newname = _convert_name(name)
    signal.name = newname
    comb = pd.concat([other, signal], axis=1)
    comb = comb.dropna()
    comb=comb.groupby('month_end').filter(lambda df:len(df)>CROSS_LEAST) #trick: at least 300 stocks in each month
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
        ts.to_pickle(os.path.join(DIR_ANALYSE,'fm',name+'.pkl'))
        print(name)


def calculate_fmt():
    names=get_playing_indicators()
    # fns=os.listdir(os.path.join(DIR_DM_NORMALIZED))
    # names=[fn[:-4] for fn in fns]
    print('total',len(names))
    checked=[fn[:-4] for fn in os.listdir(os.path.join(DIR_ANALYSE,'fm'))]
    names=[n for n in names if n not in checked]#fixme:
    print('remainder',len(names))

    multi_process(_get_fmt, names, 20,size_in_each_group=500) #fixme:

    # for name in names:
    #     _get_fmt(name)


def _get_t(fn):
    return pd.read_pickle(os.path.join(DIR_ANALYSE,'fm',fn))

def get_fmt():
    # _get_tvalue=lambda x:pd.read_pickle(os.path.join(DIR_CHORDIA,'fm',x)).at[_convert_name(x[:-4]),'tvalue']
    fns=os.listdir(os.path.join(DIR_ANALYSE,'fm'))
    df=pd.concat(multi_process(_get_t,fns),axis=1).T
    df.to_pickle(os.path.join(DIR_ANALYSE,'fmt.pkl'))

def analyze_fmt():
    df=pd.read_pickle(os.path.join(DIR_ANALYSE,'fmt.pkl'))
    df[abs(df)>3].notnull().sum(axis=1)


def main():
    calculate_fmt()
    get_fmt()

if __name__ == '__main__':
    main()

