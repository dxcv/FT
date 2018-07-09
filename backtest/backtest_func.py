import os

import numpy as np
import pandas as pd
import warnings

from config import DIR_BACKTEST

warnings.filterwarnings('ignore', category=FutureWarning)
import backtest.base_func as bf

(close_price_none,
 close_price_post,
 open_price_post,
 vwap_post,
 stocks_opened,
 stocks_need_drop) = bf.read_trade_data('close_price_none',
                                        'close_price_post',
                                        'open_price_post',
                                        'vwap_post',
                                        'stocks_opened',
                                        'stocks_need_drop',
                                        data_path=os.path.join(DIR_BACKTEST,'backtest_data.h5'))

close_price_backtest = close_price_post
trade_date = close_price_backtest.index.to_series()
last_close_price_none = close_price_none.shift(1)

buy_stocks_price = close_price_backtest
sell_stocks_price = close_price_backtest


# buy_stocks_price = vwap_post
# sell_stocks_price = vwap_post


def backtest(date_range, signal, buy_commission=2e-4, sell_commission=2e-4,
             *args):
    tax_ratio = 0.001  # 印花税
    capital = 100000000  # 虚拟资本

    trade_returns = pd.Series(index=date_range)  # 组合收益率
    turnover_rates = pd.Series(index=date_range)  # 换手率
    positions_record = []  # 市值仓位记录
    shares_record = []  # 份额仓位记录
    transactions_record = []  # 交易记录

    new_hold_shares = pd.Series()
    today_market_value = capital
    for day in date_range:
        # 记录昨日持仓权重和股数，以及昨日收盘价
        last_hold_shares = new_hold_shares
        last_market_value = today_market_value

        # 读取今日市场数据
        AStocks_close_price = close_price_backtest.loc[day]

        # 读取今日交易价格数据
        AStocks_buy_price = buy_stocks_price.loc[day]
        AStocks_sell_price = sell_stocks_price.loc[day]

        # 获取今日目标持仓
        target_list = signal_to_targetlist(day, signal, last_hold_shares.copy(),
                                           args[0], args[1], args[2])

        # 生成初始订单列表
        order_list = target_list.add(-last_hold_shares, fill_value=0)
        order_list = order_list.round(6)

        invariable_list = target_list[order_list == 0]  # 持仓不动订单
        sell_list = -order_list[order_list < 0]  # 需卖出订单
        buy_list = order_list[order_list > 0]  # 需买入订单

        # 计算今日卖出金额
        sell_shares = last_hold_shares[sell_list.index]
        sell_values = sell_shares * AStocks_sell_price[sell_shares.index]
        sell_values_sum = sum(sell_values)
        sell_fee = sell_values_sum * (tax_ratio + sell_commission)

        # 计算今日买入金额
        buy_fee = (sell_values_sum - sell_fee) * buy_commission
        if last_hold_shares.empty:
            buy_values_sum = capital
        else:
            buy_values_sum = sell_values_sum - sell_fee - buy_fee

        buy_list /= buy_list.sum()  # 买入的各股相对权重
        buy_values = buy_values_sum * buy_list
        buy_shares = buy_values / AStocks_buy_price[buy_values.index] #review:不考虑数值调整吗？比如最小交易数量为一手 100股

        invariable_shares = last_hold_shares[#review: 只要在invariable_list中的就不变了吗？不改变权重吗？
            invariable_list.index]  # 获取今日持仓不动股票份额
        new_hold_shares = pd.concat(
            [invariable_shares, buy_shares])  # 更新今日最新持仓股票份额

        new_hold_values = new_hold_shares * AStocks_close_price[
            new_hold_shares.index]

        today_market_value = new_hold_values.sum()  # 计算今日持仓总市值
        if (last_hold_shares.empty) | (last_market_value == 0):
            today_return = 0
        else:
            today_return = today_market_value / last_market_value - 1  # 计算今日收益率

        if last_market_value != 0:
            turnover = sell_values_sum / last_market_value  # 换手率
        else:
            turnover = np.nan
        trade_returns[day] = today_return
        turnover_rates[day] = turnover
        positions_record.append(new_hold_values.copy())
        shares_record.append(new_hold_shares.copy())
        transac = pd.concat([-sell_shares, buy_shares])
        transac.name = day
        transactions_record.append(transac)

    return trade_returns, turnover_rates, positions_record, shares_record, transactions_record


def signal_to_effectivelist(day, signal, effective_number, transform_mode):
    effective_list = signal.loc[day]
    need_drop_stocks = set()
    if day in stocks_need_drop.index:
        need_drop_stocks = stocks_need_drop[day]
        if isinstance(need_drop_stocks, str):
            need_drop_stocks = set([need_drop_stocks])
        else:
            need_drop_stocks = set(need_drop_stocks)
    effective_list.drop(list(need_drop_stocks), inplace=True)
    effective_list.replace([np.inf, -np.inf], np.nan, inplace=True)
    effective_list.dropna(inplace=True)
    effective_list = effective_list.sort_values(ascending=False)[
                     :effective_number] # keep the largest n (effective_number) stock

    if transform_mode == 1:# handle the abnormal value
        mean_ = effective_list.mean()
        std_ = effective_list.std()
        left_3std = mean_ - 3 * std_
        right_3std = mean_ + 3 * std_
        normal = effective_list[
            (effective_list >= left_3std) & (effective_list <= right_3std)]
        normal_min = normal.min()
        normal_max = normal.max()
        effective_list[effective_list < left_3std] = normal_min
        effective_list[effective_list > right_3std] = normal_max
    elif transform_mode == 2:
        effective_list -= effective_list.mean()
        k = 1 / effective_list.abs().sum()
        effective_list *= k
        effective_list = effective_list.apply(np.arctan)
    elif transform_mode == 3:
        effective_list[:] = 1

    if transform_mode in [0, 1, 2]:
        min_ = effective_list.min()
        max_ = effective_list.max()
        if min_ < 0:
            effective_list -= min_#此处得出的最小值为0
            effective_list += abs(min_)# 平移一下，避免0值
    #             effective_list += max_

    effective_list = effective_list[effective_list != 0]
    return effective_list


def effectivelist_to_targetlist(day, effective_list, last_hold_list,
                                target_number):
    if not effective_list.empty:
        price = close_price_none.loc[day]
        last_price = last_close_price_none.loc[day]
        buy_limited_AStocks = price[price > 1.095 * last_price].index # 涨停的股票
        sell_limited_AStocks = price[price < 0.905 * last_price].index# 跌停的股票
        AStocks_opened = (stocks_opened.loc[day] == 1)
        opened_AStocks = AStocks_opened[AStocks_opened].index

        coincident_list = last_hold_list[
            last_hold_list.index.isin(effective_list.index)]
        selling_stocks = set(last_hold_list.index) - set(
            coincident_list.index) - set(buy_limited_AStocks) #不卖涨停的股票，即便根据信号来说该卖
        sellable_stocks = (selling_stocks & set(opened_AStocks)) - set(
            sell_limited_AStocks)
        hold_list = last_hold_list.drop(list(sellable_stocks))

        if len(hold_list) < target_number:
            buyable_stocks = (
                        (set(effective_list.index) & set(opened_AStocks)) -
                        set(buy_limited_AStocks) - set(sell_limited_AStocks)) #不买跌停股，即便根据信号来说该买
            effective_list = effective_list[list(buyable_stocks)]
            alternative_list = effective_list.drop(hold_list.index,
                                                   errors='ignore')
            alternative_list.sort_values(ascending=False, inplace=True)

            complement_number = target_number - len(hold_list)
            buy_list = alternative_list[:complement_number]
            if len(buy_list) != 0:
                if buy_list.sum() != 0:
                    buy_list /= buy_list.sum()
                else:
                    buy_list[:] = 1 / len(buy_list)
                buy_list = buy_list.round(8)
                buy_list = buy_list[buy_list != 0]

            if len(buy_list) < complement_number:
                sellable_list = last_hold_list[sellable_stocks].sort_values()
                substitute_list = sellable_list[
                                  :complement_number - len(buy_list)]
            else:
                substitute_list = pd.Series()
            target_list = hold_list.append([buy_list, substitute_list])

        else:
            target_list = last_hold_list

    else:
        target_list = last_hold_list

    return target_list


def signal_to_targetlist(day, signal, last_hold_list, effective_number,
                         target_number, transform_mode):
    if day in signal.index:
        effective_list = signal_to_effectivelist(day, signal, effective_number,
                                                 transform_mode)
        target_list = effectivelist_to_targetlist(day, effective_list,
                                                  last_hold_list, target_number)
    else:
        target_list = last_hold_list

    return target_list
