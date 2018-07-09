# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-06  17:53
# NAME:FT-a.py
from singleFactor.check import plot_beta_t_ic, plot_layer_analysis

directory=r'E:\a\1-ratio_chg-net_profit-net_profit_excl_min_int_inc'
beta_t_ic=pd.read_csv(os.path.join(directory,'beta_t_ic'))

fig_beta_t_ic = plot_beta_t_ic(beta_t_ic)

fig_g = plot_layer_analysis(g_ret, g_ret_des, cover_rate)



fig_beta_t_ic.savefig(os.path.join(directory, 'fig_beta_t_ic.png'))
fig_g.savefig(os.path.join(directory, 'fig_g.png'))