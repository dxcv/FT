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
from matplotlib.colors import ListedColormap

#TODO:make sure that all the assets share the same window
#TODO:if the model contains the underline factor of the testing asset,skip it



def _pricing_with_grs(arg):
    '''get the GRS with one model and one set of assets'''
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
    '''
    get GRS for each possible three factor models with market factor and two return facotrs

    Returns:

    '''
    nmodels=[i[:-4] for i in os.listdir(os.path.join(DIR_KOGAN,'models','3'))]
    nassets=[i[:-4] for i in os.listdir(os.path.join(DIR_KOGAN,'assets','eq'))]
    arg_list=[]
    for nm in nmodels:
        for na in nassets:
            arg_list.append((nm,na))
    pp=multi_task(_pricing_with_grs, arg_list)
    grs_factor=pd.DataFrame(np.array(pp).reshape(len(nmodels),len(nassets)),
                      index=nmodels,columns=nassets)

    get_names=lambda x:[x.split('___')[0],x.split('___')[-1]]

    for ind in grs_factor.index:
        grs_factor.loc[ind,get_names(ind)]=np.nan

    grs_factor.T.to_csv(os.path.join(DIR_KOGAN_RESULT,'grs_factor.csv'))

def grs_pca_model():
    '''
    get GRS for pca model with respect to the 21 sets of assets constructed by sorting on characteristics
    Returns:

    '''
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

def pricing_with_grs_all():
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
    '''generate all the possible 3-factor models and 4-factor models'''
    directory=os.path.join(DIR_KOGAN,'port_ret','eq')
    names=[fn[:-4] for fn in os.listdir(directory)]
    names_list=list(itertools.combinations(names,2))+list(itertools.combinations(names,3))
    multi_task(_generate_models, names_list)

def get_raw_factors():
    '''get high-minu-low factors'''
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

def match_based_on_alpha_pvalue():
    '''use all the possible 3-factor models to pricing the factor returns and use pvalue of
    the alpha as the matching criterion. Count the number each of the 3-factor model can
    match with respect to the 21 factor returns'''

    modelnames=os.listdir(os.path.join(DIR_KOGAN,'models','3'))
    factors=get_raw_factors()

    _matched_l=[]
    _names_l=[]
    for mname in modelnames:
        model=pd.read_pickle(os.path.join(DIR_KOGAN,'models','3',mname))
        comb=pd.concat([model,factors],axis=1)
        matched_num=0
        for f in factors.columns:
            if f not in model.columns:
                sub=comb[model.columns.tolist()+[f]].dropna()
                Y=sub[f]
                X=sm.add_constant(sub[model.columns])
                r=sm.OLS(Y,X).fit()
                p=r.pvalues.loc['const']
                if p>CRITICAL:
                    matched_num+=1
        print(mname)
        _matched_l.append(matched_num)
        _names_l.append(mname)

    index=pd.MultiIndex.from_tuples((mn[:-4].split('___') for mn in _names_l))
    matched=pd.Series(_matched_l,index=index)
    return matched

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
    '''Table 2:variation explained- principle-component analysis of return factors'''
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
    '''Table 3: principle-component factor loadings'''
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
        plt.savefig(os.path.join(DIR_KOGAN_RESULT,'factor_correlation_heatmap.pdf'))

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

def get_performance_distribution(weight=1):
    '''
    figure2: factor model performance distribution
    Args:
        weight:

    Returns:

    '''
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

    plt.plot(distribution['percentile'],distribution['performance'])
    for n in specials:
        x,y=distribution.at[n,'percentile'],distribution.at[n,'performance']
        plt.plot(x,y,'--bo',label=n)
        plt.annotate(n,xy=(x,y))
    # plt.show()
    figname='factor model performance distribution_{}.png'.format('Characteristic Freq' if weight else 'Equal-weighted')
    plt.savefig(os.path.join(DIR_KOGAN_RESULT,figname))
    plt.close()

# get_performance_distribution(0)
# get_performance_distribution(1)


def get_factor_model_performance():
    '''
    figure 3: Factor Model Performance
    Returns:

    '''
    grs_factor = pd.read_csv(os.path.join(DIR_KOGAN_RESULT, 'grs_factor.csv'),
                             index_col=0)

    grs_factor[grs_factor >= CRITICAL]=1
    grs_factor[grs_factor < CRITICAL]=0

    models_ascending=grs_factor.sum().sort_values(kind='mergesort').index
    factors_ascending=grs_factor.sum(axis=1).sort_values(kind='mergesort').index

    df=grs_factor.reindex(index=factors_ascending,columns=models_ascending)
    df=df.fillna(-1)
    df.columns=np.linspace(0,1,df.shape[1])


    sns.set(font_scale=0.8)
    # cmap is now a list of colors
    cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=3)

    # Create two appropriately sized subplots
    grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}

    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws,figsize=(df.shape[1]/10,df.shape[0]/10))

    ax = sns.heatmap(df, ax=ax, cbar_ax=cbar_ax, cmap=ListedColormap(cmap),
                     linewidths=0.01, linecolor='lightgray',
                     cbar_kws={'orientation': 'vertical'})

    # Customize tick marks and positions
    cbar_ax.set_yticklabels(['None','<{}'.format(CRITICAL),'>={}'.format(CRITICAL)])
    cbar_ax.yaxis.set_ticks([-1,0,1])

    # X - Y axis labels
    ax.set_ylabel('FROM')
    ax.set_xlabel('TO')

    # Rotate tick labels
    locs, labels = plt.xticks()

    plt.setp(labels, rotation=0)
    locs, labels = plt.yticks()
    plt.setp(labels, rotation=0)

    plt.savefig(os.path.join(DIR_KOGAN_RESULT,'factor model performance.pdf'))



# if __name__ == '__main__':
#     pricing_with_grs_all()