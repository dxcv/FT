# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  10:54
# NAME:FT_hp-pricing_assets.py


import os
import pandas as pd
import numpy as np
from empirical.build_models import get_pca_model
from empirical.config import DIR_KOGAN, DIR_KOGAN_RESULT
from empirical.utils import GRS_test, run_GRS_test
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

    grs_factor.to_csv(os.path.join(DIR_KOGAN_RESULT,'grs_factor.csv'))

    # return df_p

def grs_pca_model():
    model=get_pca_model()
    fn_assets = os.listdir(os.path.join(DIR_KOGAN, 'assets', 'eq'))
    _ps=[]
    for fa in fn_assets:
        asset = pd.read_pickle(os.path.join(DIR_KOGAN, 'assets', 'eq', fa))
        _,p=run_GRS_test(model,asset)
        _ps.append(p)

    grs_pca=pd.Series(_ps,index=fn_assets)
    grs_pca.to_csv(os.path.join(DIR_KOGAN_RESULT,'grs_pca.csv'))

    # return grs_pca

def debug():
    df_p=grs_factor_model()
    grsp=grs_pca_model()

    grsp[grsp>0.05].notnull().sum()
    df_p[df_p>0.05].notnull().sum(axis=1)


    return

if __name__ == '__main__':
    grs_factor_model()
    grs_pca_model()






# if __name__ == '__main__':
#     grs_factor_model()
