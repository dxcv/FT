# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-09-15  19:29
# NAME:FT_hp-5 pricing_hedged_factors.py

from empirical.bootstrap import pricing_assets
from empirical.config_ep import DIR_CHORDIA
from empirical.data_mining.dm_api import get_raw_factors
from empirical.get_basedata import BENCHS, get_benchmark
from empirical.utils import align_index
import os
import pandas as pd


def pricing_all_factors(bench_name):
    # raw_factors=pd.read_pickle(os.path.join(DIR_DM,'raw_factors.pkl'))
    # return pd.read_pickle(path)
    raw_factors=get_raw_factors()
    bench_name, assets=align_index(bench_name, raw_factors)
    result=pricing_assets(bench_name, assets)
    s=result['alpha_t'].sort_values()
    return s

def get_alpha_t_for_all_bm():
    for bname in BENCHS:
        print(bname)
        bench=get_benchmark(bname)
        # if isinstance(bench, pd.Series):#capmM
        #     bench = bench.to_frame()
        s=pricing_all_factors(bench)
        s.to_pickle(os.path.join(DIR_CHORDIA,f'at_{bname}.pkl'))#alpha t value

def combine_at():
    at = pd.concat(
        [pd.read_pickle(os.path.join(DIR_CHORDIA, f'at_{bench}.pkl'))
         for bench in BENCHS], axis=1,keys=BENCHS,sort=True)
    at.to_pickle(os.path.join(DIR_CHORDIA,'at.pkl'))


def main():
    get_alpha_t_for_all_bm()
    combine_at()


if __name__ == '__main__':
    main()
