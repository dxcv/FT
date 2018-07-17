import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def portfolio_performance(portfolio_return, benchmark_index):
    return_free = 0.0              # 无风险利率估算为3%
    
    portfolio_value = (portfolio_return + 1).cumprod()
    benchmark_value = benchmark_index / benchmark_index[0]
    benchmark_return = benchmark_value.pct_change()
    
    portfolio_annualized_return = portfolio_value[-1] ** (252/len(portfolio_value)) - 1
    benchmark_annualized_return = benchmark_value[-1] ** (252/len(benchmark_value)) - 1
    
    beta = portfolio_return.cov(benchmark_return) / benchmark_return.var()
    alpha = (portfolio_annualized_return - return_free) - beta * (benchmark_annualized_return - return_free)

    volatility = portfolio_return.std() * (252 ** 0.5)
    sharp_ratio = (portfolio_annualized_return - return_free) / volatility

    track_err_std = (portfolio_return - beta * benchmark_return).std() * (252 ** 0.5)
    information_ratio = alpha / track_err_std

    max_drawdown = 1 - min(portfolio_value / np.maximum.accumulate(portfolio_value.fillna(-np.inf)))
    
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
    return perf


def get_hedged_returns(portfolio_returns, benchmark_index, hedged_period):
    corrected_ratio = 1.0
    
    benchmark_returns = benchmark_index.pct_change()
    benchmark_returns[0] = 0
    
    temp_hedged_value = []
    last_net_val = 1
    for i in range(0, len(portfolio_returns), hedged_period):
        temp_unit_val = (portfolio_returns[i : (i+hedged_period)] + 1).cumprod()
        temp_bch_unit_val = (benchmark_returns[i : (i+hedged_period)] + 1).cumprod()

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
    
    font = mpl.font_manager.FontProperties(fname=os.path.dirname(__file__)+'/plot_font.otf')
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
        (0.00, label_height, value_height, u"回测收益", "{0:.2%}".format(perf["portfolio_total_return"]), black, red),
        (0.00, label_height2, value_height2, u"年化收益", "{0:.2%}".format(perf["portfolio_annualized_return"]), black, red),        
        
        (0.085, label_height, value_height, "Alpha", "{0:.2%}".format(perf["alpha"]), black, black),
        (0.085, label_height2, value_height2, "Beta", "{0:.3f}".format(perf["beta"]), black, black),
        
        (0.16, label_height, value_height, "夏普比率", "{0:.3f}".format(perf["sharp_ratio"]), black, black),
        (0.16, label_height2, value_height2, "信息比率", "{0:.3f}".format(perf["information_ratio"]), black, black),
        
        (0.24, label_height, value_height, "最大回撤", "{0:.2%}".format(perf["max_drawdown"]), black, black),
        (0.24, label_height2, value_height2, "年化波动率", "{0:.2%}".format(perf["volatility"]), black, black),        
        
        
        (0.37, label_height, value_height, u"对冲收益", "{0:.2%}".format(hedged_perf["portfolio_total_return"]), black, blue),
        (0.37, label_height2, value_height2, u"对冲年化收益", "{0:.2%}".format(hedged_perf["portfolio_annualized_return"]), black, blue),        
        
        (0.46, label_height, value_height, "Alpha", "{0:.2%}".format(hedged_perf["alpha"]), black, black),
        (0.46, label_height2, value_height2, "Beta", "{0:.3f}".format(hedged_perf["beta"]), black, black),
        
        (0.53, label_height, value_height, "夏普比率", "{0:.3f}".format(hedged_perf["sharp_ratio"]), black, black),
        (0.53, label_height2, value_height2, "信息比率", "{0:.3f}".format(hedged_perf["information_ratio"]), black, black),
        
        (0.61, label_height, value_height, "最大回撤", "{0:.2%}".format(hedged_perf["max_drawdown"]), black, black),
        (0.61, label_height2, value_height2, "年化波动率", "{0:.2%}".format(hedged_perf["volatility"]), black, black),
        
        
        (0.75, label_height, value_height, "收益回撤比", "{0:.3f} / {1:.3f}".format(perf["return_down_ration"], 
                                                                                   hedged_perf["return_down_ration"]), black, black),
        (0.75, label_height2, value_height2, "平均换手率", "{0:.3f}".format(portfolio_turnover.mean()), black, black),
        
        
        (0.90, label_height, value_height, u"基准收益", "{0:.2%}".format(perf["benchmark_total_return"]), black, black),
        (0.90, label_height2, value_height2, u"基准年化收益", "{0:.2%}".format(perf["benchmark_annualized_return"]), black, black),
    ]
    
    ax1 = fig.add_subplot(gs[:3])
    ax1.axis("off")
    for x, y1, y2, label, value, label_color, value_color in text_data:
        ax1.text(x, y1, label, color=label_color, fontproperties=font, fontsize=font_size)
        ax1.text(x, y2, value, color=value_color, fontproperties=font, fontsize=value_font_size)
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
    return_diff_positive = pd.Series(np.zeros(len(return_diff)), index=return_diff.index)
    return_diff_negative = pd.Series(np.zeros(len(return_diff)), index=return_diff.index)
    return_diff_positive.where(return_diff < 0, return_diff, inplace=True)
    return_diff_negative.where(return_diff > 0, return_diff, inplace=True)
    ax3.fill_between(return_diff_positive.index, return_diff_positive.values, color=red, label='高于基准')
    ax3.fill_between(return_diff_negative.index, return_diff_negative.values, color=green, label='低于基准')
    ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2%}'))
    ax3.legend(loc='upper left', prop=font)
    ax3.spines['top'].set_color('none')
    ax3.spines['right'].set_color('none')
    
    
    ax4 = fig.add_subplot(gs[15:])    
    ax4.fill_between(portfolio_turnover.index, portfolio_turnover.values, 
                 portfolio_turnover.mean(), color='gray', lw=1, label=u'换手率')    
    ax4.legend(loc='best', prop=font)
    ax4.spines['top'].set_color('none')
    ax4.spines['right'].set_color('none')
    
#    plt.savefig(title + '.png', bbox_inches='tight')
#    plt.show()
    if fig_handler:
        return fig
    
    
def portfolio_year_performance(portfolio_return, benchmark_return, portfolio_turnover_rate):
    return_free = 0.03              # 无风险利率估算为3%
    
    year = pd.Series(portfolio_return.index.year)
    year = year.drop_duplicates().values
    year = [str(y) for y in year]
    perf_year = pd.DataFrame(np.nan, index=year, 
                             columns=['return', 'excess_return', 'benchmark_return', 'max_drawdown', 
                                      'volatility', 'sharp_ratio', 'turnover_rate'])
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
            ret = value[-1] ** (252/day_number) - 1
            bch_ret = bch_value[-1] ** (252/day_number) - 1
        exs_ret = ret - bch_ret

        sharp = (ret - return_free) / volat       
        turnover = portfolio_turnover_rate[y].mean()
        perf_year.loc[y] = [ret, exs_ret, bch_ret, max_down, volat, sharp, turnover]
    return perf_year


def hedged_year_performance(hedged_return, benchmark_return):
    return_free = 0.03              # 无风险利率估算为3%
    
    year = pd.Series(hedged_return.index.year)
    year = year.drop_duplicates().values
    year = [str(y) for y in year]
    perf_year = pd.DataFrame(np.nan, index=year, 
                             columns=['return', 'benchmark_return', 'max_drawdown', 
                                      'volatility', 'sharp_ratio', 'ret_draw_ratio'])
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
            ret = value[-1] ** (252/day_number) - 1
            bch_ret = bch_value[-1] ** (252/day_number) - 1
        exs_ret = ret - bch_ret

        sharp = (ret - return_free) / volat       
        ret_draw_ratio = ret / max_down
        perf_year.loc[y] = [ret, bch_ret, max_down, volat, sharp, ret_draw_ratio]
    return perf_year


def format_year_performance(returns, benchmark_ind, turnover_rate, title):
    from IPython.display import display
    benchmark_returns = benchmark_ind.pct_change()
    benchmark_returns[0] = 0
    perf_year = portfolio_year_performance(returns, benchmark_returns, turnover_rate)
    perf_year.columns = [u'年度收益', u'超额收益', u'基准收益', u'最大回撤', 
                         u'波动率', u'夏普比率', u'换手率']
    perf_year.index.name = u'年份'
    format_funcs = {u'年度收益': '{:.2%}'.format, u'超额收益': '{:.2%}'.format, 
                    u'基准收益': '{:.2%}'.format, u'最大回撤': '{:.2%}'.format, 
                    u'波动率': '{:.2%}'.format, u'夏普比率': '{:.2f}'.format, 
                    u'换手率': '{:.3f}'.format}
    perf_year = perf_year.transform(format_funcs)
    print (' '*int((66 - len(title))/2) + title)
    # display(perf_year)
    
    
def format_hedged_year_performance(returns, benchmark_ind, title):
    from IPython.display import display
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
    print (' '*int((62 - len(title))/2) + title)
    display(perf_year)