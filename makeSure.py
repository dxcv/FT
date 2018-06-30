# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-30  00:09
# NAME:FT-makeSure.py


def check_format_for_indicators(df):
    if df.index.names != ['stkcd','trd_dt']:
        raise ValueError('{} is not allowed'.format(df.index.names))
    elif df.shape[1]>1:
        raise ValueError('df should be with one column')


