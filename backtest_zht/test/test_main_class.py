# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-29  13:09
# NAME:FT_hp-test_main_class.py

import pandas as pd
from backtest_zht.main_class import Backtest
from backtest_zht.main import run_backtest
from config import DIR_TMP
import os


signal=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\singleFactor\mixed_signal\300_10_return_std_ratio.pkl')

name='test1'


config={
        'effective_number': 150,
        'target_number': 100,
        'signal_to_weight_mode': 3, #等权重
        # 'decay_num': 1,　＃TODO：　used to smooth the signal
        # 'delay_num': 1,
        'hedged_period': 60, #trick: 股指的rebalance时间窗口，也可以考虑使用风险敞口大小来作为relance与否的依据
        'buy_commission': 2e-3,#tric: 实际应该是2e-4左右，我们这里用的是2e-3是把冲击成本也加在里边了
        'sell_commission': 2e-3,
        'tax_ratio':0.001,# 印花税
        'capital':10000000,#虚拟资本,没考虑股指期货所需要的资金
        }

Backtest(signal,name,os.path.join(DIR_TMP,name),start='2009',config=config)




