# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-07  17:07
# NAME:FT-main.py



import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from backtest.backtest_func import trade_date, backtest, buy_stocks_price, \
    sell_stocks_price, close_price_backtest
from config import DIR_BACKTEST

sns.set_style('white', {'axes.linewidth': 1.0, 'axes.edgecolor': '.8'})
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from collections import OrderedDict
import statsmodels.api as sm
# import pyfolio as pf
import alphalens as alen

import sys

# sys.path.append('./lib/')
# from backtest.backtest_func import *
from backtest.plot_performance import portfolio_performance,get_hedged_returns,plot_portfolio_performance,format_year_performance,format_hedged_year_performance
import backtest.base_func as bf
global_settings = {'effective_number': 200,
                   'target_number': 100,
                   'transform_mode': 3, #等权重
                   # 'decay_num': 1,　＃TODO：　used　ｔｏ　ｓｍｏｏｔｈ　ｔｈｅ　ｓｉｇｎａｌ
                   # 'delay_num': 1,
                   'hedged_period': 60, #trick: 股指的rebalance时间窗口，也可以考虑使用风险敞口大小来作为relance与否的依据
                   'buy_commission': 2e-3,
                   'sell_commission': 2e-3
                   }



zz500, = bf.read_trade_data('zz500', data_path=os.path.join(DIR_BACKTEST,'backtest_data.h5'))

benchmark_returns_zz500 = zz500.pct_change()
benchmark_returns_zz500.name = 'benchmark'

def quick(signal,fig_title,start=None, end=None):
    global global_settings
    hedged_period = global_settings['hedged_period']

    if not start:
        start=trade_date[0]
    if not end:
        end=trade_date[-1]

    date_range = trade_date[start: end]
    benchmark = zz500[start: end]

    trade_returns, turnover_rates, positions_record, shares_record, transactions_record = backtest(
        date_range, signal,
        global_settings['buy_commission'], global_settings['sell_commission'],
        global_settings['effective_number'], global_settings['target_number'],
        global_settings['transform_mode'])

    turnover_rates[0] = np.nan
    positions_record = pd.concat(positions_record, keys=date_range,
                                 names=['tradeDate'])
    shares_record = pd.concat(shares_record, keys=date_range,
                              names=['tradeDate'])
    transactions_record = pd.concat(transactions_record,
                                    axis=1).stack().swaplevel().sort_index(
        level=0)

    # for quantopia format
    positions_record = positions_record.unstack()
    positions_record['cash'] = np.nan

    shares_record = shares_record.unstack()

    txn_date = transactions_record.index.get_level_values(0)
    txn_symbol = transactions_record.index.get_level_values(1)
    txn_amount = transactions_record.values
    txn_price = np.zeros(len(txn_date))
    for i in range(len(txn_date)):
        if txn_amount[i] >= 0:
            txn_price[i] = buy_stocks_price.loc[txn_date[i], txn_symbol[i]]
        else:
            txn_price[i] = sell_stocks_price.loc[txn_date[i], txn_symbol[i]]
    transactions_record = pd.DataFrame(
        {'amount': txn_amount, 'price': txn_price, 'symbol': txn_symbol},
        index=txn_date)

    perf = portfolio_performance(trade_returns, benchmark)
    hedged_returns = get_hedged_returns(trade_returns, benchmark, hedged_period)
    hedged_perf = portfolio_performance(hedged_returns, benchmark)
    fig=plot_portfolio_performance(trade_returns, turnover_rates, hedged_returns,
                                   benchmark, perf, hedged_perf, fig_title, fig_handler=True)
    # format_year_performance(trade_returns, benchmark, turnover_rates,
    #                         fig_title)

    # format_hedged_year_performance(hedged_returns, benchmark,
    #                                fig_title + '_hedged')

    results = OrderedDict({
        'trade_returns': trade_returns,
        'turnover_rates': turnover_rates,
        'positions_record': positions_record,
        'shares_record': shares_record,
        'transactions_record': transactions_record,
        'hedged_returns': hedged_returns,
        'hedged_perf':hedged_perf
    })
    return results,fig


#TODO: 应该提供几种接口: 1. signal， 2. 股票，
#TODO:分别测算每年的对冲收益率，因为实际投资的时候不可能投资５年，所以ｃｕｍｐｒｏｄ实际会夸大效果
