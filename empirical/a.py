# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-31  20:19
# NAME:FT_hp-a.py


import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

sm.OLS(y, X).fit().summary()
# sm.OLS(y1,X).fit().summary()



# params=np.linalg.pinv(X.T @ X) @ X.T @ y
# predictions=X@params
#
# MSE=(sum((y-predictions)**2))/(X.shape[0]-X.shape[1])
# var_b=MSE*np.linalg.inv(X.T @ X).diagonal()
# sd_b=np.sqrt(var_b)
# ts_b=params/sd_b
# p_values=[2*(1-stats.t.cdf(np.abs(i),X.shape[0]-1)) for i in ts_b]
#
#
# sd_b = np.round(sd_b,3)
# ts_b = np.round(ts_b,3)
# p_values = np.round(p_values,3)
# params = np.round(params,4)
#
# myDF3 = pd.DataFrame()
# myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
# print(myDF3)
#
#
#
#
#

# params=np.linalg.pinv(X.T @ X) @ X.T @ y1
# predictions=X@params
#
# MSE=(sum((y1-predictions)**2))/(X.shape[0]-X.shape[1])
#
#
# var_b=MSE*np.linalg.inv(X.T @ X).diagonal()
#
# sd_b=np.sqrt(var_b)
# ts_b=params/sd_b
# p_values=[2*(1-stats.t.cdf(np.abs(i),X.shape[0]-1)) for i in ts_b]
#
# p_values



from line_profiler import LineProfiler
import random

def do_stuff(numbers):
    s = sum(numbers)
    l = [numbers[i]/43 for i in range(len(numbers))]
    m = ['hello'+str(numbers[i]) for i in range(len(numbers))]

numbers = [random.randint(1,100) for i in range(1000)]
lp = LineProfiler()
lp_wrapper = lp(do_stuff)
lp_wrapper(numbers)
lp.print_stats()