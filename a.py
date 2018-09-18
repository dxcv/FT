# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-08-27  21:42
# NAME:FT_hp-a.py
import pandas as pd
import os

from config import DIR_TMP

from itertools import islice, chain


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

it=range(33)
test=chunks(it,5)

for i,chunk in enumerate(test):
    for c in chunk:
        print(i,c)



