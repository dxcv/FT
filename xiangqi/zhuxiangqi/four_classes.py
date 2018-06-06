# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:04:27 2018

@author: XQZhu
"""

import pandas as pd
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib')
import class_test as ct

class_names = ['Value', 'Momentum', 'Volatility', 'Consensus']
store = pd.HDFStore('test_data.h5')
for class_name in class_names[2:]:
    columns = store[class_name].columns.tolist()
    factor_names = [name[:-4] for name in columns]
    BTIC, IC, IC_corr, Annual, Sharpe, Rela_IR = ct.class_test(factor_names[:9], class_name)
    writer = pd.ExcelWriter(class_name + '.xlsx')
    BTIC.to_excel(writer, 'BTIC')
    IC_corr.to_excel(writer, 'IC_corr')
    Annual.to_excel(writer, 'Annual Return')
    Sharpe.to_excel(writer, 'Sharpe Ratio')
    Rela_IR.to_excel(writer, 'Relative Return IR')
    writer.save()
