# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-25  15:32
# NAME:FT_hp-config.py
from config import DIR_BACKTEST

CONFIG={
        'effective_number': 200,
        'target_number': 100,
        'signal_to_weight_mode': 3, #等权重
        # 'decay_num': 1,　＃TODO：　used to smooth the signal
        # 'delay_num': 1,
        'hedged_period': 60, #trick: 股指的rebalance时间窗口，也可以考虑使用风险敞口大小来作为relance与否的依据
        'buy_commission': 2e-3,
        'sell_commission': 2e-3,
        'tax_ratio':0.001,# 印花税
        'capital':10000000,#虚拟资本,没考虑股指期货所需要的资金
        }

