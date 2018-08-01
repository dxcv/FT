# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-25  08:54
# NAME:FT_hp-replication.py
import multiprocessing
import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from config import DIR_TMP
from sklearn.decomposition import PCA
import statsmodels.api as sm
import numpy as np
from empirical.config import NUM_FACTOR
import os
from empirical.get_basedata import get_benchmark
from empirical.config import DIR_KOGAN, DIR_KOGAN_RESULT, CRITICAL
from empirical.utils import run_GRS_test
from numpy.linalg import LinAlgError
from tools import multi_task

#TODO:make sure that all the assets share the same window
#TODO:if the model contains the underline factor of the testing asset,skip it



def one_model_one_cohort(arg):
    fm,fa=arg
    model = pd.read_pickle(os.path.join(DIR_KOGAN, 'models', '3', fm+'.pkl'))
    asset = pd.read_pickle(os.path.join(DIR_KOGAN, 'assets', 'eq', fa+'.pkl'))
    try:
        _, p = run_GRS_test(model, asset)
    except LinAlgError as e:
        # TODO: what's wrong with this situation?
        # print('{}\n\tmodel:{}\n\tasset:{}'.format(e, fm, fa))
        p = np.nan
    return p

def grs_factor_model():
    nmodels=[i[:-4] for i in os.listdir(os.path.join(DIR_KOGAN,'models','3'))]
    nassets=[i[:-4] for i in os.listdir(os.path.join(DIR_KOGAN,'assets','eq'))]
    arg_list=[]
    for nm in nmodels:
        for na in nassets:
            arg_list.append((nm,na))
    pp=multi_task(one_model_one_cohort,arg_list)
    grs_factor=pd.DataFrame(np.array(pp).reshape(len(nmodels),len(nassets)),
                      index=nmodels,columns=nassets)

    get_names=lambda x:[x.split('___')[0],x.split('___')[-1]]

    for ind in grs_factor.index:
        grs_factor.loc[ind,get_names(ind)]=np.nan

    grs_factor.T.to_csv(os.path.join(DIR_KOGAN_RESULT,'grs_factor.csv'))

    # return df_p

def grs_pca_model():
    model=get_pca_model()
    fn_assets = os.listdir(os.path.join(DIR_KOGAN, 'assets', 'eq'))
    _ps=[]
    for fa in fn_assets:
        asset = pd.read_pickle(os.path.join(DIR_KOGAN, 'assets', 'eq', fa))
        _,p=run_GRS_test(model,asset)
        _ps.append(p)

    grs_pca=pd.Series(_ps,index=[f[:-4] for f in fn_assets],name='pca').to_frame()
    grs_pca.to_csv(os.path.join(DIR_KOGAN_RESULT,'grs_pca.csv'))

    # return grs_pca

def _grs_bench(args):
    fa,bn=args
    asset=pd.read_pickle(os.path.join(DIR_KOGAN,'assets','eq',fa))
    model=get_benchmark(bn)
    if isinstance(model,pd.Series):
        model=model.to_frame()
    _,p=run_GRS_test(model,asset)
    return p


def grs_benchmodel():
    bench_names=['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M', 'ff6M']
    fn_assets = os.listdir(os.path.join(DIR_KOGAN, 'assets', 'eq'))
    args_list=[]
    for fa in fn_assets:
        for bn in bench_names:
            args_list.append((fa,bn))

    _ps=multi_task(_grs_bench,args_list)

    grs_benchmark=pd.DataFrame(np.array(_ps).reshape((len(fn_assets),len(bench_names))),
                               index=[f[:-4] for f in fn_assets], columns=bench_names)
    grs_benchmark.to_csv(os.path.join(DIR_KOGAN_RESULT,'grs_benchmark.csv'))

def grs_all():
    grs_factor_model()
    grs_pca_model()
    grs_benchmodel()



def _generate_models(names):
    directory=os.path.join(DIR_KOGAN,'port_ret','eq')
    factors=[]
    for name in names:
        factor=pd.read_pickle(os.path.join(directory,name+'.pkl'))['tb']
        factor.name=name
        factors.append(factor)

    rpM=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata','rpM.pkl'))
    comb=pd.concat(factors+[rpM],axis=1).dropna()
    comb.to_pickle(os.path.join(DIR_KOGAN,'models',str(len(names)+1),'___'.join(names)+'.pkl'))

def build_models():
    directory=os.path.join(DIR_KOGAN,'port_ret','eq')
    names=[fn[:-4] for fn in os.listdir(directory)]
    names_list=list(itertools.combinations(names,2))+list(itertools.combinations(names,3))
    multi_task(_generate_models, names_list)

def get_raw_factors():
    directory = os.path.join(DIR_KOGAN, 'port_ret', 'eq')
    fns = os.listdir(directory)

    dfs = []
    for fn in fns:
        df = pd.read_pickle(os.path.join(directory, fn))['tb']
        dfs.append(df)

    raw_factors = pd.concat(dfs, axis=1, keys=[fn[:-4] for fn in fns])
    raw_factors = raw_factors.dropna(thresh=int(raw_factors.shape[1] * 0.8))
    raw_factors = raw_factors.fillna(0)
    return raw_factors

def get_pca_factors(n=3):
    raw_factors=get_raw_factors()
    X=raw_factors.values
    pca=PCA(n_components=n)
    pca_factors=pd.DataFrame(pca.fit_transform(X),index=raw_factors.index,
                             columns=['pca{}'.format(i) for i in range(1,n+1)])
    return pca_factors

def get_pca_model(n=NUM_FACTOR):
    pca_factors=get_pca_factors(n=n-1)
    rpM=pd.read_pickle(os.path.join(DIR_KOGAN,'basedata','rpM.pkl'))
    pca_model=pd.concat([rpM,pca_factors],axis=1).dropna()
    return pca_model

def get_table2():
    comb=get_raw_factors()
    X=comb.values
    rs=[]
    for n in range(1,X.shape[1]+1):
        pca=PCA(n_components=n)
        pca.fit(X)
        # a=pca.fit_transform(X)
        ratio=np.sum(pca.explained_variance_ratio_)
        rs.append(ratio)

    variation_explained=pd.Series(rs,index=range(1,X.shape[1]+1))
    variation_explained.plot().get_figure().show()

def get_table3():
    raw_factors=get_raw_factors()
    X=raw_factors.values
    pca=PCA(n_components=NUM_FACTOR)
    pca.fit(X)

    factor_loading=pd.DataFrame(pca.components_.T,index=raw_factors.columns,columns=['pca{}'.format(i) for i in range(1,NUM_FACTOR+1)])

def get_heatmap():
    '''figure1: factor correlation'''
    raw_factors=get_raw_factors()
    pca_model4=get_pca_model(4)
    # pca_factors=get_pca_factors(n=3)

    factors=pd.concat([raw_factors,pca_model4],axis=1).dropna()


    corr=factors.corr()

    with sns.axes_style('white'):
        ax=sns.heatmap(corr,linewidth=0.5,cmap='YlGnBu')
        plt.show()

def spanning_regression():
    '''table4: factor regression on the principal-component model'''
    pca_model3=get_pca_model(n=3)
    raw_factors=get_raw_factors()

    comb=pd.concat([pca_model3,raw_factors],axis=1).dropna()

    values=[]
    for col in raw_factors.columns:
        Y=comb[col].values
        X=comb[pca_model3.columns].values
        X=sm.add_constant(X)
        model=sm.OLS(Y,X)
        r=model.fit()
        values.append((r.params[0], r.tvalues[0], r.rsquared_adj))
    spanning_result=pd.DataFrame(values,index=raw_factors.columns,columns=['alpha','t','adj_r2'])


def get_performance(weight=0):
    '''
    p>CRITICAL  denotes the model can not pricing the assets perfectly

    Args:
        weight:{0,1},0 denote equal weighed,1 denotes Characteristic Matching Frequency
    Returns:

    '''
    grs_factor=pd.read_csv(os.path.join(DIR_KOGAN_RESULT,'grs_factor.csv'),index_col=0)
    grs_pca=pd.read_csv(os.path.join(DIR_KOGAN_RESULT,'grs_pca.csv'),index_col=0)
    grs_benchmark=pd.read_csv(os.path.join(DIR_KOGAN_RESULT,'grs_benchmark.csv'),index_col=0)

    grs_factor[grs_factor >= CRITICAL] = 1
    grs_factor[grs_factor < CRITICAL] = 0
    if weight==0:
        #performace1: equal weight
        perf=grs_factor.sum(axis=1)/(grs_factor.shape[1]-NUM_FACTOR+1)
    else:
        weight=1-grs_factor.sum()/grs_factor.shape[0]
        # weight=1-grs_model[grs_model>CRITICAL].notnull().sum()/grs_model.shape[0]
        perf=(grs_factor*weight).sum(axis=1)/(grs_factor.shape[1]-NUM_FACTOR+1)
    perf=perf.sort_values(ascending=False,kind='mergesort')
    return perf

weight=0

grs_factor = pd.read_csv(os.path.join(DIR_KOGAN_RESULT, 'grs_factor.csv'),
                         index_col=0)
grs_pca = pd.read_csv(os.path.join(DIR_KOGAN_RESULT, 'grs_pca.csv'),
                      index_col=0)
grs_benchmark = pd.read_csv(
    os.path.join(DIR_KOGAN_RESULT, 'grs_benchmark.csv'), index_col=0)

comb=pd.concat([grs_factor,grs_pca,grs_benchmark],axis=1,sort=False)
comb[comb>=CRITICAL]=1
comb[comb<CRITICAL]=0
#fixme: it's unjustice to compare three factor model with ff6
if weight==0:
    perf_factor_model=comb[grs_factor.columns].sum()/(comb.shape[0]-NUM_FACTOR+1)
    perf_pca=comb[grs_pca.columns].sum()/comb.shape[0]
    perf_benchmark=comb[grs_benchmark.columns].sum()/comb.shape[0]
    perf=pd.concat([perf_factor_model,perf_pca,perf_benchmark]).sort_values(ascending=False,kind='mergesort')
else:
    freq_weight=1-comb[grs_factor.columns].sum(axis=1)/grs_factor.shape[1]
    perf_factor_model=(comb[grs_factor.columns].T*freq_weight).T.sum()/(comb.shape[0]-NUM_FACTOR+1)
    perf_pca=(comb[grs_pca.columns].T*freq_weight).T.sum()/comb.shape[0]
    perf_benchmark=(comb[grs_benchmark.columns].T*freq_weight).T.sum()/comb.shape[0]
    perf=pd.concat([perf_factor_model,perf_pca,perf_benchmark]).sort_values(ascending=False,kind='mergesort')

distribution=perf.sort_values().to_frame()
distribution.columns=['performance']
distribution['percentile']=np.arange(1,distribution.shape[0]+1)/distribution.shape[0]

distribution.set_index('percentile')['performance'].plot().get_figure().show()
specials=grs_pca.columns.tolist()+grs_benchmark.columns.tolist()
plt.plot(distribution['percentile'],distribution['performance'],markevery=distribution.index.isin(specials))



for n in specials:
    plt.plot(distribution.at[n,'percentile'],distribution.at[n,'performance'],'g*')


plt.savefig(os.path.join(DIR_TMP,'distribution.png'))


# if __name__ == '__main__':
#     grs_all()
