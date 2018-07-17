# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-20  17:30
# NAME:FT-financial.py
import multiprocessing
from functools import reduce

from config import SINGLE_D_INDICATOR, FORWARD_LIMIT_Q, FORWARD_TRADING_DAY
from data.dataApi import get_dataspace, read_local, read_from_sql
import os
import re

from singleFactor.operators import *
from tools import daily2monthly

def save_indicator(df,name):
    df.to_pickle(os.path.join(SINGLE_D_INDICATOR,name+'.pkl'))

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

def quarterly_to_daily(df,name,duplicates='last'):
    '''

    Args:
        df:DataFrame, contains ['stkcd','report_period','trd_dt'] in its columns
          or index
        name:
        duplicates:

    Returns:

    '''

    td=read_from_sql('trade_date','ftresearch')['dates'].values
    #trick:There are some duplicates (different report_period) even with same stkcd and trd_dt
    # df=df.reset_index().sort_values(['stkcd','trd_dt','report_period'])
    # df = df[~df.duplicated(subset=['stkcd', 'trd_dt'], keep='last')]

    #TODO: mean or last for duplicates
    daily=pd.pivot_table(df,values=name,index='trd_dt',columns='stkcd',aggfunc=duplicates)

    daily=daily.reindex(td)#review:
    daily.index.name='trd_dt'
    daily=daily.ffill(limit=FORWARD_TRADING_DAY)# debug: 向前填充最最多400个交易日,年度频率的数据，400 问题不大，但是对于月频的数据，最多向前填充400个交易日肯定是有问题的
    daily=daily.dropna(how='all') #trick
    return daily

def parse_a_row(s):
    name = '__'.join([s['type'], s['name']])
    print(name)
    eq_x = s['numerator']
    eq_y = s['denominator']
    func = s['function']
    if eq_y==1:
        vars=parse_vars(eq_x)
        df=get_dataspace(vars)
        df['x']=parse_equation(eq_x)
        kwarg = parse_args(s['kwarg']) if isinstance(s['kwarg'], str) else None
        if kwarg:
            df[name]=eval(func)(df['x'],**kwarg)
        else:
            df[name]=eval(func)(df['x'])
    else:
        vars = parse_vars(eq_x) + parse_vars(eq_y)
        df = get_dataspace(vars)
        df['x'] = parse_equation(eq_x)
        df['y'] = parse_equation(eq_y)
        kwarg = parse_args(s['kwarg']) if isinstance(s['kwarg'], str) else None
        if kwarg:
            df[name] = eval(func)(df, 'x', 'y', **kwarg)
        else:
            df[name] = eval(func)(df, 'x', 'y')

    df=df.dropna(subset=[name])
    if df.shape[0]>0:
        daily=quarterly_to_daily(df,name)
        save_indicator(daily, name)
    else:
        pass


# path=r'indicators.xlsx'
# df=pd.read_excel(path,sheet_name='equation',index_col=0)
# s=df.loc[25]
# parse_a_row(s)

def debug():
    path = r'indicators.xlsx'
    df = pd.read_excel(path, sheet_name='equation', index_col=0)
    parse_a_row(df.loc[19])


def cal_sheet_equation():
    path=r'indicators.xlsx'
    df=pd.read_excel(path,sheet_name='equation',index_col=0)
    pool=multiprocessing.Pool(10)
    pool.map(parse_a_row,(s for _,s in df.iterrows()))
    # for _,s in df.iterrows():
    #     parse_a_row(s)

def cal_sheet_growth():
    func_id={'x_pct_chg':'pct',
             'x_history_compound_growth':'hcg',
             'x_history_std':'std'}

    path = 'indicators.xlsx'
    df = pd.read_excel(path, sheet_name='growth', index_col=0)
    indicators=df['indicator']
    for _,s in df[['function','kwarg']].dropna().iterrows():
        func=s['function']
        kwarg=parse_args(s['kwarg'])
        for indicator in indicators:
            name='G_{}_{}__{}'.format(func_id[func],kwarg['q'],indicator)
            df=get_dataspace(indicator)
            df[name]=eval(func)(df[indicator],**kwarg)
            df = df.dropna(subset=[name])
            if df.shape[0] > 0:
                daily = quarterly_to_daily(df, name)
                save_indicator(daily, name)
            print(func,indicator,kwarg['q'])


# if __name__ == '__main__':
    # cal_sheet_equation()
    # cal_sheet_growth()
