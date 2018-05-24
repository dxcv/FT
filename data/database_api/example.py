# 请留意使用help()查看函数内部的帮助文档。

import pandas as pd
from data import database_api as dbi

# 获取股票数据
stocks_data = dbi.get_stocks_data('equity_selected_trading_data', ['open', 'close', 'volume'], 
                                  '2009-01-01', '2010-12-31')

# 获取指数数据
index_data = dbi.get_index_data('zz500', '2009-01-01', '2010-12-31')

# 获取期货数据
futures_data = dbi.get_futures_data('commodityitems_daily', ['OP', 'CP', 'CQ'], ['A', 'RB'], 
                                    '2009-01-01', '2010-12-31')

# 若要将数据保存到本地，请分别按如下方式调用
with pd.HDFStore('database_stocks_data.h5') as file:
    stocks_data = dbi.get_stocks_data('equity_selected_trading_data', ['open', 'close', 'volume'], 
                                      '2009-01-01', '2010-12-31', 
                                      store=True, hdf_file=file, data_name='trade_price')						   
						   
with pd.HDFStore('database_stocks_data.h5') as file:
    index_data = dbi.get_index_data('zz500', '2009-01-01', '2010-12-31', 
                                    store=True, hdf_file=file, data_name='zz500')						  
					
with pd.HDFStore('database_futures_data.h5') as file:
    futures_data = dbi.get_futures_data('commodityitems_daily', ['OP', 'CP', 'CQ'], ['A', 'RB'], 
                                        '2009-01-01', '2010-12-31', 
                                        store=True, hdf_file=file, data_name='A_RB_price')