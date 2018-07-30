# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-29  12:37
# NAME:FT_hp-main_class.py

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import backtest_zht.base_func as bf
from backtest_zht.config import DIR_BACKTEST,CONFIG
from collections import OrderedDict

from config import DIR_TMP

DEFAULT_CONFIG={
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

zz500, = bf.read_trade_data('zz500', data_path=os.path.join(DIR_BACKTEST,'backtest_data.h5'))

benchmark_returns_zz500 = zz500.pct_change()
benchmark_returns_zz500.name = 'benchmark'


close_price_backtest = close_price_post
trade_date = close_price_backtest.index.to_series()
last_close_price_none = close_price_none.shift(1)

buy_stocks_price = close_price_backtest
sell_stocks_price = close_price_backtest


def portfolio_performance(portfolio_return, benchmark_index):
    return_free = 0.0  # 无风险利率估算为3%

    portfolio_value = (portfolio_return + 1).cumprod()
    benchmark_value = benchmark_index / benchmark_index[0]
    benchmark_return = benchmark_value.pct_change()

    portfolio_annualized_return = portfolio_value[-1] ** (
    252 / len(portfolio_value)) - 1
    benchmark_annualized_return = benchmark_value[-1] ** (
    252 / len(benchmark_value)) - 1

    beta = portfolio_return.cov(benchmark_return) / benchmark_return.var()
    alpha = (portfolio_annualized_return - return_free) - beta * (
    benchmark_annualized_return - return_free)

    volatility = portfolio_return.std() * (252 ** 0.5)
    sharp_ratio = (portfolio_annualized_return - return_free) / volatility

    track_err_std = (portfolio_return - beta * benchmark_return).std() * (
    252 ** 0.5)
    information_ratio = alpha / track_err_std

    max_drawdown = 1 - min(portfolio_value / np.maximum.accumulate(
        portfolio_value.fillna(-np.inf)))

    perf = {
        'portfolio_total_return': portfolio_value[-1] - 1,
        'portfolio_annualized_return': portfolio_annualized_return,
        'benchmark_total_return': benchmark_value[-1] - 1,
        'benchmark_annualized_return': benchmark_annualized_return,
        'beta': beta,
        'alpha': alpha,
        'volatility': volatility,
        'sharp_ratio': sharp_ratio,
        'information_ratio': information_ratio,
        'max_drawdown': max_drawdown,
        'return_down_ration': portfolio_annualized_return / max_drawdown
    }
    return pd.Series(perf)

def get_hedged_returns(portfolio_returns, benchmark_index):
    corrected_ratio = 1.0

    benchmark_returns = benchmark_index.pct_change()
    benchmark_returns[0] = 0

    temp_hedged_value = []
    last_net_val = 1
    for i in range(0, len(portfolio_returns), CONFIG['hedged_period']):
        temp_unit_val = (
        portfolio_returns[i: (i + CONFIG['hedged_period'])] + 1).cumprod()
        temp_bch_unit_val = (
        benchmark_returns[i: (i + CONFIG['hedged_period'])] + 1).cumprod()

        temp_val = (temp_unit_val - temp_bch_unit_val + 1) * last_net_val
        temp_hedged_value.append(temp_val)
        last_net_val = temp_val[-1]

    hedged_value = pd.concat(temp_hedged_value)
    hedged_value = 1 + (hedged_value - 1) * corrected_ratio
    hedged_returns = hedged_value.pct_change()
    hedged_returns[0] = 0
    return hedged_returns

def plot_portfolio_performance(portfolio_return, portfolio_turnover,
                               hedged_return, benchmark_index, perf,
                               hedged_perf, title, fig_handler=False):
    sns.set_style('white', {'axes.linewidth': 1.0, 'axes.edgecolor': '.8'})

    portfolio_value = (portfolio_return + 1).cumprod()
    hedged_value = (hedged_return + 1).cumprod()
    benchmark_value = benchmark_index / benchmark_index[0]
    benchmark_return = benchmark_value.pct_change()
    benchmark_return[0] = 0

    #     mpl.rcParams['font.family'] = 'sans-serif'
    #     mpl.rcParams['font.sans-serif'] = [
    #         u'Microsoft Yahei',
    #         u'SimHei',
    #         u'sans-serif']
    #     mpl.rcParams['axes.unicode_minus'] = False

    font = mpl.font_manager.FontProperties(
        fname=os.path.dirname(__file__) + '/plot_font.otf')
    #    font = mpl.font_manager.FontProperties(fname='plot_font.otf')
    red = "#aa4643"
    green = '#156b09'
    blue = "#4572a7"
    black = "#000000"

    fig = plt.figure(figsize=(18, 14))
    gs = mpl.gridspec.GridSpec(18, 1)

    x_date = portfolio_value.index
    # x_data = np.arange(0, len(x_date))

    font_size = 12
    value_font_size = 14
    label_height, value_height = 0.75, 0.55
    label_height2, value_height2 = 0.30, 0.10

    text_data = [
        (0.00, label_height, value_height, u"回测收益",
         "{0:.2%}".format(perf["portfolio_total_return"]), black, red),
        (0.00, label_height2, value_height2, u"年化收益",
         "{0:.2%}".format(perf["portfolio_annualized_return"]), black, red),

        (0.085, label_height, value_height, "Alpha",
         "{0:.2%}".format(perf["alpha"]), black, black),
        (0.085, label_height2, value_height2, "Beta",
         "{0:.3f}".format(perf["beta"]), black, black),

        (0.16, label_height, value_height, "夏普比率",
         "{0:.3f}".format(perf["sharp_ratio"]), black, black),
        (0.16, label_height2, value_height2, "信息比率",
         "{0:.3f}".format(perf["information_ratio"]), black, black),

        (0.24, label_height, value_height, "最大回撤",
         "{0:.2%}".format(perf["max_drawdown"]), black, black),
        (0.24, label_height2, value_height2, "年化波动率",
         "{0:.2%}".format(perf["volatility"]), black, black),

        (0.37, label_height, value_height, u"对冲收益",
         "{0:.2%}".format(hedged_perf["portfolio_total_return"]), black, blue),
        (0.37, label_height2, value_height2, u"对冲年化收益",
         "{0:.2%}".format(hedged_perf["portfolio_annualized_return"]), black,
         blue),

        (0.46, label_height, value_height, "Alpha",
         "{0:.2%}".format(hedged_perf["alpha"]), black, black),
        (0.46, label_height2, value_height2, "Beta",
         "{0:.3f}".format(hedged_perf["beta"]), black, black),

        (0.53, label_height, value_height, "夏普比率",
         "{0:.3f}".format(hedged_perf["sharp_ratio"]), black, black),
        (0.53, label_height2, value_height2, "信息比率",
         "{0:.3f}".format(hedged_perf["information_ratio"]), black, black),

        (0.61, label_height, value_height, "最大回撤",
         "{0:.2%}".format(hedged_perf["max_drawdown"]), black, black),
        (0.61, label_height2, value_height2, "年化波动率",
         "{0:.2%}".format(hedged_perf["volatility"]), black, black),

        (0.75, label_height, value_height, "收益回撤比",
         "{0:.3f} / {1:.3f}".format(perf["return_down_ration"],
                                    hedged_perf["return_down_ration"]), black,
         black),
        (0.75, label_height2, value_height2, "平均换手率",
         "{0:.3f}".format(portfolio_turnover.mean()), black, black),

        (0.90, label_height, value_height, u"基准收益",
         "{0:.2%}".format(perf["benchmark_total_return"]), black, black),
        (0.90, label_height2, value_height2, u"基准年化收益",
         "{0:.2%}".format(perf["benchmark_annualized_return"]), black, black),
    ]

    ax1 = fig.add_subplot(gs[:3])
    ax1.axis("off")
    for x, y1, y2, label, value, label_color, value_color in text_data:
        ax1.text(x, y1, label, color=label_color, fontproperties=font,
                 fontsize=font_size)
        ax1.text(x, y2, value, color=value_color, fontproperties=font,
                 fontsize=value_font_size)
    ax1.set_title(title, fontproperties=font, fontsize=20)

    ax2 = fig.add_subplot(gs[4:10])
    ax2.plot(portfolio_value - 1, c=red, lw=2, label=u'策略')
    ax2.plot(hedged_value - 1, c=blue, lw=2, label=u'对冲')
    ax2.plot(benchmark_value - 1, c='gray', lw=2, label=u'基准')

    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2%}'))
    ax2.legend(loc='best', prop=font)
    ax2.grid()

    ax3 = fig.add_subplot(gs[11:14])
    return_diff = portfolio_return - benchmark_return
    return_diff_positive = pd.Series(np.zeros(len(return_diff)),
                                     index=return_diff.index)
    return_diff_negative = pd.Series(np.zeros(len(return_diff)),
                                     index=return_diff.index)
    return_diff_positive.where(return_diff < 0, return_diff, inplace=True)
    return_diff_negative.where(return_diff > 0, return_diff, inplace=True)
    ax3.fill_between(return_diff_positive.index, return_diff_positive.values,
                     color=red, label='高于基准')
    ax3.fill_between(return_diff_negative.index, return_diff_negative.values,
                     color=green, label='低于基准')
    ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2%}'))
    ax3.legend(loc='upper left', prop=font)
    ax3.spines['top'].set_color('none')
    ax3.spines['right'].set_color('none')

    ax4 = fig.add_subplot(gs[15:])
    ax4.fill_between(portfolio_turnover.index, portfolio_turnover.values,
                     portfolio_turnover.mean(), color='gray', lw=1,
                     label=u'换手率')
    ax4.legend(loc='best', prop=font)
    ax4.spines['top'].set_color('none')
    ax4.spines['right'].set_color('none')

    #    plt.savefig(title + '.png', bbox_inches='tight')
    #    plt.show()
    if fig_handler:
        plt.close()
        return fig

def portfolio_year_performance(portfolio_return, benchmark_return,
                               portfolio_turnover_rate):
    return_free = 0.03  # 无风险利率估算为3%

    year = pd.Series(portfolio_return.index.year)
    year = year.drop_duplicates().values
    year = [str(y) for y in year]
    perf_year = pd.DataFrame(np.nan, index=year,
                             columns=['return', 'excess_return',
                                      'benchmark_return', 'max_drawdown',
                                      'volatility', 'sharp_ratio',
                                      'turnover_rate'])
    for y in year:
        value = (portfolio_return[y] + 1).cumprod()
        bch_value = (benchmark_return[y] + 1).cumprod()

        max_down = 1 - min(value[y] / np.maximum.accumulate(value))
        volat = portfolio_return[y].std() * (252 ** 0.5)

        day_number = len(value[y])
        if day_number > 230:
            ret = value[-1] - 1
            bch_ret = bch_value[-1] - 1
        else:
            ret = value[-1] ** (252 / day_number) - 1
            bch_ret = bch_value[-1] ** (252 / day_number) - 1
        exs_ret = ret - bch_ret

        sharp = (ret - return_free) / volat
        turnover = portfolio_turnover_rate[y].mean()
        perf_year.loc[y] = [ret, exs_ret, bch_ret, max_down, volat, sharp,
                            turnover]
    return perf_year

def hedged_year_performance(hedged_return, benchmark_return):
    return_free = 0.03  # 无风险利率估算为3%

    year = pd.Series(hedged_return.index.year)
    year = year.drop_duplicates().values
    year = [str(y) for y in year]
    perf_year = pd.DataFrame(np.nan, index=year,
                             columns=['return', 'benchmark_return',
                                      'max_drawdown',
                                      'volatility', 'sharp_ratio',
                                      'ret_draw_ratio'])
    for y in year:
        value = (hedged_return[y] + 1).cumprod()
        bch_value = (benchmark_return[y] + 1).cumprod()

        max_down = 1 - min(value[y] / np.maximum.accumulate(value))
        volat = hedged_return[y].std() * (252 ** 0.5)

        day_number = len(value[y])
        if day_number > 230:
            ret = value[-1] - 1
            bch_ret = bch_value[-1] - 1
        else:
            ret = value[-1] ** (252 / day_number) - 1
            bch_ret = bch_value[-1] ** (252 / day_number) - 1
        exs_ret = ret - bch_ret

        sharp = (ret - return_free) / volat
        ret_draw_ratio = ret / max_down
        perf_year.loc[y] = [ret, bch_ret, max_down, volat, sharp,
                            ret_draw_ratio]
    return perf_year

def format_year_performance(returns, benchmark_ind, turnover_rate):
    # from IPython.display import display
    benchmark_returns = benchmark_ind.pct_change()
    benchmark_returns[0] = 0
    perf_year = portfolio_year_performance(returns, benchmark_returns,
                                           turnover_rate)
    perf_year.columns = [u'年度收益', u'超额收益', u'基准收益', u'最大回撤',
                         u'波动率', u'夏普比率', u'换手率']
    perf_year.index.name = u'年份'
    format_funcs = {u'年度收益': '{:.2%}'.format, u'超额收益': '{:.2%}'.format,
                    u'基准收益': '{:.2%}'.format, u'最大回撤': '{:.2%}'.format,
                    u'波动率': '{:.2%}'.format, u'夏普比率': '{:.2f}'.format,
                    u'换手率': '{:.3f}'.format}
    perf_year = perf_year.transform(format_funcs)
    # print(' ' * int((66 - len(title)) / 2) + title)
    # display(perf_year)
    return perf_year

def format_hedged_year_performance(returns, benchmark_ind):
    # from IPython.display import display
    benchmark_returns = benchmark_ind.pct_change()
    benchmark_returns[0] = 0
    perf_year = hedged_year_performance(returns, benchmark_returns)
    perf_year.columns = [u'年度收益', u'基准收益', u'最大回撤',
                         u'波动率', u'夏普比率', u'收益回撤比']
    perf_year.index.name = u'年份'
    format_funcs = {u'年度收益': '{:.2%}'.format, u'基准收益': '{:.2%}'.format,
                    u'最大回撤': '{:.2%}'.format, u'波动率': '{:.2%}'.format,
                    u'夏普比率': '{:.2f}'.format, u'收益回撤比': '{:.3f}'.format}
    perf_year = perf_year.transform(format_funcs)
    # print(' ' * int((62 - len(title)) / 2) + title)
    # display(perf_year)
    return perf_year


class Backtest:
    def __init__(self,signal, name, directory, start=None, end=None,config=DEFAULT_CONFIG):
        self.signal=signal
        self.name=name
        self.directory=directory
        self.start=start if start else self.signal.index[0]
        self.end=end if end else self.signal.index[-1]
        self.config=config
        self.date_range=trade_date[start:end]
        self.benchmark=zz500[self.start:self.end]
        self.run()

    def signal_to_effectivelist(self,day, signal):
        effective_list = signal.loc[day]
        need_drop_stocks = set()
        if day in stocks_need_drop.index:
            need_drop_stocks = stocks_need_drop[day]
            if isinstance(need_drop_stocks, str):
                # need_drop_stocks = set([need_drop_stocks])
                need_drop_stocks = {need_drop_stocks}
            else:
                need_drop_stocks = set(need_drop_stocks)
        effective_list.drop(list(need_drop_stocks), inplace=True,
                            errors='ignore')
        effective_list.replace([np.inf, -np.inf], np.nan, inplace=True)
        effective_list.dropna(inplace=True)
        #fixme: 在从Series中截取数据的时候，如果我们要从series中根据value截取n
        #fixme:个最大的值， 在每个value的不相等的情况下，s.sort_values（ascending=False)[:n] 没问题，
        #但是如果value 不全相等的时候，可能会出现每次回测的时候选取的股票都不同的情况。所以，为了保证相同的signal产生相同的
        #回测结果，最好是保证每次选取的股票都相同。所以，这里应该加入一个选股过程，对于value相同的股票做一个排序。
        #为了方便，我们这里直接省略了这个步骤，直接用股票代码排序，s.sort_index().sort_values(ascending=False)[:n]以保证回测结果的唯一性。但是，这样会导致每次遇到这种
        #情况，我们都会优先选择股票代码较小的股票。（后边所有用Series排序选股票的方式都有这个问题）

        effective_list = effective_list.sort_index().sort_values(ascending=False)[:CONFIG[
            'effective_number']]  # keep the largest n (effective_number) stock

        if CONFIG['signal_to_weight_mode'] == 1:  # handle the abnormal value
            # if transform_mode == 1:# handle the abnormal value
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
        elif CONFIG['signal_to_weight_mode'] == 2:  # transform with arctan
            effective_list -= effective_list.mean()
            k = 1 / effective_list.abs().sum()
            effective_list *= k
            effective_list = effective_list.apply(np.arctan)
        elif CONFIG['signal_to_weight_mode'] == 3:  # equally weighted
            effective_list[:] = 1

        if CONFIG['signal_to_weight_mode'] in [0, 1, 2]:
            min_ = effective_list.min()
            max_ = effective_list.max()
            if min_ < 0:
                effective_list -= min_  # 此处得出的最小值为0
                effective_list += abs(min_)  # 平移一下，避免0值
        # effective_list += max_

        effective_list = effective_list[effective_list != 0]
        return effective_list

    def effectivelist_to_targetlist(self,day, effective_list, last_hold_list):
        if not effective_list.empty:
            price = close_price_none.loc[day]
            last_price = last_close_price_none.loc[
                day]  # trick: we had to use the non adjusted price
            buy_limited_AStocks = price[
                price > 1.095 * last_price].index  # 涨停的股票
            sell_limited_AStocks = price[
                price < 0.905 * last_price].index  # 跌停的股票
            AStocks_opened = (
            stocks_opened.loc[day] == 1)  # review: simplify this two rows
            opened_AStocks = AStocks_opened[AStocks_opened].index
            # 用effective_list 这种方式去确定调不调仓有点naive,也可以考虑用比例，实际情况可能需考虑持仓和市场行情
            # 不需要调仓的股票
            coincident_list = last_hold_list[
                last_hold_list.index.isin(
                    effective_list.index)]  # trick： effective_list cover more stock than target_number, and we only sell those stocks that are no longer covered by effective list.
            selling_stocks = set(last_hold_list.index) - set(
                coincident_list.index) - set(
                buy_limited_AStocks)  # 不卖涨停的股票，即便根据信号来说该卖
            sellable_stocks = (selling_stocks & set(opened_AStocks)) - set(
                sell_limited_AStocks)
            hold_list = last_hold_list.drop(list(sellable_stocks))
            if len(hold_list) < CONFIG['target_number']:
                buyable_stocks = (
                    (set(effective_list.index) & set(opened_AStocks)) -
                    set(buy_limited_AStocks) - set(
                        sell_limited_AStocks))  # 不买跌停股，即便根据信号来说该买
                effective_list = effective_list[list(buyable_stocks)]
                alternative_list = effective_list.drop(hold_list.index,
                                                       errors='ignore')  # rebalance时候考虑的股票池
                #fixme: 选股问题，同上
                alternative_list.sort_index().sort_values(ascending=False, inplace=True)

                complement_number = CONFIG['target_number'] - len(
                    hold_list)  # 这种定量持有100股的持仓方式可以优化一下，不一定非要持有100股
                buy_list = alternative_list[:complement_number]
                if len(buy_list) != 0:
                    if buy_list.sum() != 0:
                        buy_list /= buy_list.sum()
                    else:
                        buy_list[:] = 1 / len(buy_list)
                    buy_list = buy_list.round(8)  # 有时候有些数值特别小，这种情况就直接省略对应的股票
                    buy_list = buy_list[buy_list != 0]

                if len(buy_list) < complement_number:  # trick:由于上一步删除了数值较小的股票，为了达到持有100只股票的目的，这里应该继续追加股票，但是，更好的方式是少卖一些股票，这样可以减少卖出和买入交易成本
                    #fixme: 选股问题，同上,但是这里的问题是，last_hold_list中的value是上一个交易日的持有金额，而上面的value是signal值
                    sellable_list = last_hold_list[sellable_stocks].sort_index().sort_values(ascending=False)
                    substitute_list = sellable_list[:complement_number - len(buy_list)]
                else:
                    substitute_list = pd.Series()
                target_list = hold_list.append([buy_list, substitute_list])

            else:
                target_list = last_hold_list

        else:
            target_list = last_hold_list

        return target_list

    def signal_to_targetlist(self,day, signal, last_hold_list):
        if day in signal.index:
            effective_list = self.signal_to_effectivelist(day, signal)
            target_list = self.effectivelist_to_targetlist(day, effective_list,
                                                      last_hold_list)
        else:
            target_list = last_hold_list

        return target_list

    def backtest(self):
        trade_returns = pd.Series(index=self.date_range)  # 组合收益率
        turnover_rates = pd.Series(index=self.date_range)  # 换手率
        positions_record = []  # 市值仓位记录
        shares_record = []  # 份额仓位记录
        transactions_record = []  # 交易记录

        new_hold_shares = pd.Series()
        today_market_value = CONFIG['capital']
        for day in self.date_range:  # 在策略第一天运行的时候肯定不能直接这样在第一天建仓100只股票，会造成市场冲击，所以，策略在高换手率期间应该有所优化
            print('backtesting: {}'.format(day))
            # 记录昨日持仓权重和股数，以及昨日收盘价
            last_hold_shares = new_hold_shares
            last_market_value = today_market_value

            # 读取今日市场数据
            AStocks_close_price = close_price_backtest.loc[day]

            # 读取今日交易价格数据
            AStocks_buy_price = buy_stocks_price.loc[day]
            AStocks_sell_price = sell_stocks_price.loc[day]

            # 获取今日目标持仓
            target_list = self.signal_to_targetlist(day, self.signal,
                                               last_hold_shares.copy())

            # 生成初始订单列表
            order_list = target_list.add(-last_hold_shares, fill_value=0)
            order_list = order_list.round(6)  # 主要是为了剔除掉那些数值太小的股票

            invariable_list = target_list[order_list == 0]  # 持仓不动订单
            sell_list = -order_list[order_list < 0]  # 需卖出订单
            buy_list = order_list[order_list > 0]  # 需买入订单

            # 计算今日卖出金额
            sell_shares = last_hold_shares[sell_list.index]
            sell_values = sell_shares * AStocks_sell_price[
                sell_shares.index]  # 实际卖出的时候一般不会是收盘价，这里可以优化一下，交易算法可以单独一个模块来写，这里的买卖价格应该由交易算法模块传过来，而非直接使用开盘价收盘价
            sell_values_sum = sum(sell_values)
            # sell_fee = sell_values_sum * (tax_ratio + sell_commission)
            sell_fee = sell_values_sum * (
            CONFIG['tax_ratio'] + CONFIG['sell_commission'])

            if last_hold_shares.empty:
                # buy_values_sum=capital/(1+buy_commission)
                buy_values_sum = CONFIG['capital'] / (
                1 + CONFIG['buy_commission'])
            else:
                buy_values_sum = (sell_values_sum - sell_fee) / (
                1 + CONFIG['buy_commission'])
                # buy_values_sum=(sell_values_sum-sell_fee)/(1+buy_commission)

            buy_list /= buy_list.sum()  # 买入的各股相对权重
            buy_values = buy_values_sum * buy_list
            buy_shares = buy_values / AStocks_buy_price[
                buy_values.index]  # review:不考虑数值调整吗？比如最小交易数量为一手 100股

            invariable_shares = last_hold_shares[
                # review: 只要在invariable_list中的就不变了吗？不改变权重吗？ 万一某只股票持有了很久导致在总仓位中占比较高的话，应该有所调整
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
            transactions_record.append(transac)  # TODO：record transaction fees

        return trade_returns, turnover_rates, positions_record, shares_record, transactions_record

    def quick(self):

        trade_returns, turnover_rates, positions_record, shares_record, transactions_record = self.backtest()

        turnover_rates[0] = np.nan
        positions_record = pd.concat(positions_record, keys=self.date_range,
                                     names=['tradeDate'])
        shares_record = pd.concat(shares_record, keys=self.date_range,
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

        perf = portfolio_performance(trade_returns, self.benchmark)
        perf_yearly = format_year_performance(trade_returns, self.benchmark,
                                              turnover_rates)
        hedged_returns = get_hedged_returns(trade_returns, self.benchmark)
        hedged_perf = portfolio_performance(hedged_returns, self.benchmark)
        hedged_perf_yearly = format_hedged_year_performance(hedged_returns,
                                                            self.benchmark)
        self.fig = plot_portfolio_performance(trade_returns, turnover_rates,
                                         hedged_returns,
                                         self.benchmark, perf, hedged_perf,
                                         self.name, fig_handler=True)

        self.results = OrderedDict({
            'trade_returns': trade_returns,
            'turnover_rates': turnover_rates,
            'positions_record': positions_record,
            'shares_record': shares_record,
            'transactions_record': transactions_record,
            'hedged_returns': hedged_returns,
            'hedged_perf': hedged_perf,
            'perf': perf,
            'perf_yearly': perf_yearly,
            'hedged_perf_yearly': hedged_perf_yearly
        })

    def save_result(self):
        if os.path.exists(self.directory) and len(os.listdir(self.directory)) > 0:
            return  # skip
        elif not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.fig.savefig(os.path.join(self.directory, self.name + '.png'))
        self.signal.to_csv(os.path.join(self.directory,'signal.csv'))
        for k in self.results.keys():
            self.results[k].to_csv(os.path.join(self.directory, k + '.csv'))

    def run(self):
        self.quick()
        self.save_result()



signal=pd.read_pickle(r'G:\FT_Users\HTZhang\FT\tmp\signal_500_5_cumprod_ret_200.pkl')
name='a1'
Backtest(signal,'a1',os.path.join(DIR_TMP,'a1'))

