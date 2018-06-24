# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  17:30
# NAME:FT-new_calculate_by_use_excel.py
from functools import reduce

import pandas as pd
from config import SINGLE_D_INDICATOR
from data.dataApi import get_dataspace
import os
import re

from singleFactor.factors.new_operators import *


def save_indicator(df,name):
    df[['trd_dt',name]].to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))

def parse_vars(equation):
    '''
    get the involved indicators
    Args:
        equation:

    Returns:list

    '''
    equation=equation.replace(' ','')
    atoms = re.split('([+-])', equation)
    atoms = [a for a in atoms if a not in ['']]  # filter out ''
    if len(atoms)>1:
        if len(atoms)%2:# tot_assets - tot_liab
            vars = atoms[::2]
        else:# - tot_liab
            vars=atoms[1::2]
        return vars
    else:# for example, "tot_assets"
        return [equation]

def parse_args(s):
    '''

    Args:
        s:

    Returns:

    Examples
        s1='smooth=True'
        s2='smooth=True,q=12'
        print(parse_args(s1))
        print(parse_args(s2))


    '''
    s=s.replace(' ','')
    if ',' in s:
        return {ele.split('=')[0]:eval(ele.split('=')[1]) for ele in s.split(',')}
    else:
        return {s.split('=')[0]:eval(s.split('=')[1])}

def parse_equation(equation):
    '''

    Args:
        equation:

    Returns:pd.Series

    Examples:
        equation1='-tot_assets'
        equation2='-tot_assets+ tot_liab'
        equation3='tot_assets - tot_liab'
        equation4='tot_assets - tot_liab + monetary_cap'

        for equation in [equation1,equation2,equation3,equation4]:
            df=parse_equation(equation)
            print(equation)

    '''
    equation=equation.replace(' ','')
    atoms = re.split('([+-])', equation)
    atoms = [a for a in atoms if a not in ['']]  # filter out ''
    if len(atoms)>1:
        if len(atoms)%2:# tot_assets - tot_liab
            vars = atoms[::2]
            ops = ['+'] + atoms[1::2]
        else:# - tot_liab
            ops=atoms[::2]
            vars=atoms[1::2]
        if len(ops)==len(vars):
            df=get_dataspace(vars)
            items = []
            for op, var in zip(ops, vars):
                if op == '+':
                    items.append(df[var])
                else:
                    items.append(-df[var])
            x = reduce(lambda x, y: x + y, items)
            return x
        else:
            raise ValueError('Wrong equation:"{}"'.format(equation))
    else:# for example, "tot_assets"
        return get_dataspace(equation)[equation]

def cal_sheet_equation():
    path=r'D:\app\python36\zht\internship\FT\singleFactor\factors\indicators.xlsx'
    df=pd.read_excel(path,sheet_name='equation',index_col=0)
    for _,s in df.iterrows():
        name='__'.join([s['type'],s['name']])
        eq_x=s['numerator']
        eq_y=s['denominator']
        func=s['function']
        vars= parse_vars(eq_x) + parse_vars(eq_y)
        df=get_dataspace(vars)
        df['x']=parse_equation(eq_x)
        df['y']=parse_equation(eq_y)
        kwarg=parse_args(s['kwarg']) if isinstance(s['kwarg'],str) else None
        if kwarg:
            df[name]=eval(func)(df,'x','y',**kwarg)
        else:
            df[name]=eval(func)(df,'x','y')
        save_indicator(df,name)
        print(name)

def cal_sheet_growth():
    func_id={'x_pct_chg':'pct',
             'x_history_compound_growth':'hcg',
             'x_history_std':'std'}

    path = r'D:\app\python36\zht\internship\FT\singleFactor\factors\indicators.xlsx'
    df = pd.read_excel(path, sheet_name='growth', index_col=0)
    indicators=df['indicator']
    for _,s in df.dropna().iterrows():
        func=s['function']
        kwarg=parse_args(s['kwarg'])
        for indicator in indicators:
            name='G_{}_{}__{}'.format(func_id[func],kwarg['q'],indicator)
            df=get_dataspace(indicator)
            df[name]=eval(func)(df[indicator],**kwarg)
            save_indicator(df,name)
            print(func,indicator,kwarg['q'])

# if __name__ == '__main__':
#     cal_sheet_equation()
#     cal_sheet_growth()
