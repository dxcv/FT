# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-26  10:54
# NAME:FT_hp-pricing_assets.py

from config import DIR_KOGAN, DIR_TMP
import os
import pandas as pd
import numpy as np
from empirical.build_models import get_pca_model
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

def grs_all():
    fn_models = os.listdir(os.path.join(DIR_KOGAN, 'models', '3'))
    fn_assets = os.listdir(os.path.join(DIR_KOGAN, 'assets', 'eq'))

    arg_list=[]
    for fm in fn_models:
        for fa in fn_assets:
            arg_list.append((fm[:-4],fa[:-4]))
    pp=multi_task(one_model_one_cohort,arg_list)
    df_p=pd.DataFrame(np.array(pp).reshape(len(fn_models),len(fn_assets)),
                      index=fn_models,columns=fn_assets)

    df_p.to_csv(os.path.join(DIR_TMP,'df_p.csv'))

    return df_p


model=get_pca_model()
fn_assets = os.listdir(os.path.join(DIR_KOGAN, 'assets', 'eq'))
_ps=[]
for fa in fn_assets:
    asset=pd.read_pickle(os.path.join(DIR_KOGAN,'assets','eq',fa+'.pkl'))
    _,p=run_GRS_test(model,fa)
    _ps.append(p)

grsp=pd.Series(_ps,index=fn_assets)






# if __name__ == '__main__':
#     grs_all()
