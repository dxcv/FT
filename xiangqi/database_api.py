import numpy as np
import pandas as pd
import sqlalchemy


def database_connect(base_name):
    '''
    连接数据库。
    
    Parameters
    ----------
    base_name : str
        数据库名称。
    
    Returns
    ------
    sqlalchemy.engine
        适用于pandas连接数据库的配置引擎。
    
    '''
    
    addr = 'mysql+pymysql://ftresearch:FTResearch@192.168.1.140/{}?charset=utf8'.format(base_name)
    return sqlalchemy.create_engine(addr)


def read_security_database(base_name, sheet_name, security_ID_field, trade_date_field, other_field, 
                           select_securities=None, start_date=None, end_date=None):
    '''
    从数据库中读取证券相关数据。
    
    Parameters
    ----------
    base_name : str
        数据库名称。
    sheet_name : str
        数据库中所要读的数据表名称。
    security_ID_field : str
        证券代码对应的字段名。
    trade_date_field : str
        交易日期对应的字段名。
    other_field : list of str
        其它需取得的字段名列表。
    select_securities : list of str, or None
        所选证券代码列表。默认为None，表示选择所有证券。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
        
    Returns
    -------
    pd.DataFrame
        从数据库中取得的原始证券相关数据。
        
    '''
    
    other_field = ', '.join(other_field)
    if isinstance(select_securities, list):
        select_securities = ['\'{}\''.format(x) for x in select_securities]
        select_securities = '(' + ', '.join(select_securities) + ')'
    
    if (select_securities is None) and (start_date is None) and (end_date is None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name)
    elif (select_securities is None) and (start_date is not None) and (end_date is None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            WHERE {1}>='{4}'
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name, 
                       start_date)
    elif (select_securities is None) and (start_date is None) and (end_date is not None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            WHERE {1}<='{4}'
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name, 
                       end_date)
    elif (select_securities is None) and (start_date is not None) and (end_date is not None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            WHERE {1}>='{4}' AND {1}<='{5}'
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name, 
                       start_date, end_date)    
        
    elif (select_securities is not None) and (start_date is None) and (end_date is None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            WHERE {0} IN {4}
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name, 
                       select_securities)
    elif (select_securities is not None) and (start_date is not None) and (end_date is None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            WHERE ({0} IN {4}) AND {1}>='{5}'
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name, 
                       select_securities, start_date)
    elif (select_securities is not None) and (start_date is None) and (end_date is not None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            WHERE ({0} IN {4}) AND {1}<='{5}'
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name, 
                       select_securities, end_date)
    elif (select_securities is not None) and (start_date is not None) and (end_date is not None):
        query = '''
            SELECT {0}, {1}, {2}
            FROM {3} 
            WHERE ({0} IN {4}) AND {1}>='{5}' AND {1}<='{6}'
            '''.format(security_ID_field, trade_date_field, other_field, sheet_name, 
                       select_securities, start_date, end_date)
    
    connect = database_connect(base_name)
    data = pd.read_sql_query(sql=query, con=connect)
    return data
    
    
def retrieve_stocks_data(sheet_name, field, start_date=None, end_date=None, select_stocks=None, 
                         stock_ID_field='stkcd', trade_date_field='trd_dt', 
                         base_name='ftresearch'):
    '''
    从数据库中检索股票相关数据。
    
    Parameters
    ----------
    sheet_name : str
        数据表名称。
    field : list of str
        所选字段名列表。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
    select_stocks : list of str, or None
        所选股票代码列表。默认为None，表示选择所有股票。
    stock_ID_field : str
        股票代码对应的字段名。已默认配置好。
    trade_date_field : str
        交易日期对应的字段名。已默认配置好。
    base_name : str
        股票数据库名称。已默认配置好。
        
    Returns
    -------
    pd.DataFrame
        从数据库中检索出的股票数据，Index为MultiIndex,
        其中level0为交易日期，level1为股票代码。
    
    '''
                         
    if start_date is not None:
        start_date = ''.join(start_date.split('-'))
    if end_date is not None:
        end_date = ''.join(end_date.split('-'))
    
    data = read_security_database(base_name, sheet_name, stock_ID_field, trade_date_field, field, 
                                  select_stocks, start_date, end_date)
    data[trade_date_field] = pd.to_datetime(data[trade_date_field])
    data = data.set_index([trade_date_field, stock_ID_field])
    data = data.sort_index(level=0)
    return data
    
    
def get_stocks_data(sheet_name, field, start_date=None, end_date=None, 
                    select_stocks=None, store=False, hdf_file=None, data_name=None):
    '''
    股票数据获取接口。
    
    Parameters
    ----------
    sheet_name : str
        数据表名称。
    field : list of str
        所选字段名列表。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
    select_stocks : list of str, or None
        所选股票代码列表。默认为None，表示选择所有股票。
    store : bool
        True-存储数据到文件，Flase-不存储数据到文件。
    hdf_file : pd.HDFStore
        缓存文件名，以.h5作为扩展名。
    data_name : str
        数据名称，用作HDF文件中的key索引。
        
    Returns
    -------
    pd.DataFrame
        从数据库中检索出的股票数据，Index为MultiIndex,
        其中level0为交易日期，level1为股票代码。
        如果设置了store为True，则该数据也以key为名保存
        在hdf_file中。        
        
    '''        
    
    data = retrieve_stocks_data(sheet_name, field, start_date, end_date, select_stocks)    
    if store:
        with pd.HDFStore(hdf_file) as file:
            file.put(data_name, data)
    return data
    
    
def retrieve_futures_data(sheet_name, field, select_futures=None, start_date=None, end_date=None, 
                          future_ID_field='abridge', trade_date_field='dates', 
                          base_name='commodity'):
    '''
    从数据库中检索期货相关数据。
    
    Parameters
    ----------
    sheet_name : str
        数据表名称。
    field : list of str
        所选字段名列表。
    select_futures : list of str, or None
        所选期货品种代码列表。默认为None，表示选择所有期货品种。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
    future_ID_field : str
        期货品种代码对应的字段名。已默认配置好。
    trade_date_field : str
        交易日期对应的字段名。已默认配置好。
    base_name : str
        期货数据库名称。已默认配置好。
        
    Returns
    -------
    pd.DataFrame
        从数据库中检索出的期货数据，Index为MultiIndex,
        其中level0为交易日期，level1为期货代码。
    
    '''
                         
    data = read_security_database(base_name, sheet_name, future_ID_field, trade_date_field, ['code']+field, 
                                  select_futures, start_date, end_date)
    data = data.set_index([future_ID_field, trade_date_field])
    data = data.sort_index(level=0)
    return data
    
    
def get_futures_data(sheet_name, field, select_futures=None, start_date=None, end_date=None, 
                     store=False, hdf_file=None, data_name=None):
    '''
    期货数据获取接口。
    
    Parameters
    ----------
    sheet_name : str
        数据表名称。
    field : list of str
        所选字段名列表。
    select_futures : list of str, or None
        所选期货品种代码列表。默认为None，表示选择所有期货品种。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
    store : bool
        True-存储数据到文件，Flase-不存储数据到文件。
    hdf_file : pd.HDFStore
        缓存文件名，以.h5作为扩展名。
    data_name : str
        数据名称，用作HDF文件中的key索引。
        
    Returns
    -------
    pd.DataFrame
        从数据库中检索出的期货数据，Index为MultiIndex,
        其中level0为交易日期，level1为期货品种代码。
        如果设置了store为True，则该数据也以key为名保存
        在hdf_file中。        
        
    '''
                     
    data = retrieve_futures_data(sheet_name, field, select_futures, start_date, end_date)
    if store:
        with pd.HDFStore(hdf_file) as file:
            file.put(data_name, data)
    return data
    
    
def read_index_database(base_name, sheet_name, trade_date_field, index_names, 
                        start_date=None, end_date=None):
    '''
    从数据库中读取指数相关数据。
    
    Parameters
    ----------
    base_name : str
        数据库名称。
    sheet_name : str
        数据库中所要读的数据表名称。
    trade_date_field : str
        交易日期对应的字段名。
    index_names : list of str
        所需指数名称列表。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
    
    Returns
    -------
    pd.DataFrame
        从数据库中取得的原始指数相关数据。

    '''
    
    if isinstance(index_names, list):
        index_names = ', '.join(index_names)
        
    if (start_date is None) and (end_date is None):
        query = '''
            SELECT {0}, {1} 
            FROM {2}
            '''.format(trade_date_field, index_names, sheet_name)
    elif (start_date is not None) and (end_date is None):
        query = '''
            SELECT {0}, {1} 
            FROM {2}
            WHERE {0}>='{3}'
            '''.format(trade_date_field, index_names, sheet_name, start_date)
    elif (start_date is None) and (end_date is not None):
        query = '''
            SELECT {0}, {1} 
            FROM {2} 
            WHERE {0}<='{3}'
            '''.format(trade_date_field, index_names, sheet_name, end_date)
    elif (start_date is not None) and (end_date is not None):
        query = '''
            SELECT {0}, {1} 
            FROM {2} 
            WHERE {0}>='{3}' AND {0}<='{4}'
            '''.format(trade_date_field, index_names, sheet_name, start_date, end_date)
    
    connect = database_connect(base_name)
    data = pd.read_sql_query(sql=query, con=connect)
    return data
    
    
def retrieve_index_data(index_names, start_date=None, end_date=None, trade_date_field='trd_dt', 
                        sheet_name='equity_selected_indice_ir', base_name='ftresearch'):
    '''
    从数据库中检索指数相关数据。
    
    Parameters
    ----------
    index_names : list of str
        所需指数名称列表。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
    trade_date_field : str
        交易日期对应的字段名。已默认配置好。
    sheet_name : str
        数据库中所要读的数据表名称。已默认配置好。
    base_name : str
        股票数据库名称。已默认配置好。
    
    Returns
    -------
    pd.DataFrame
        从数据库中检索出的指数数据，Index为交易日期。
    
    '''

    if start_date is not None:
        start_date = ''.join(start_date.split('-'))
    if end_date is not None:
        end_date = ''.join(end_date.split('-'))
    
    data = read_index_database(base_name, sheet_name, trade_date_field, index_names, 
                               start_date, end_date)
    data[trade_date_field] = pd.to_datetime(data[trade_date_field])
    data = data.set_index(trade_date_field)
    return data
    
    
def get_index_data(index_names, start_date=None, end_date=None, 
                   store=False, hdf_file=None, data_name=None):
    '''
    指数数据获取接口。
    
    Parameters
    ----------
    index_names : list of str
        所需指数名称列表。
    start_date : str, or None
        所取数据的起始日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最早日期。
    end_date : str, or None
        所取数据的截至日期，格式为YYYY-MM-DD。默认为
        None，起始日期为数据库中可用的最新日期。
    store : bool
        True-存储数据到文件，Flase-不存储数据到文件。
    hdf_file : pd.HDFStore
        缓存文件名，以.h5作为扩展名。
    data_name : str
        数据名称，用作HDF文件中的key索引。
        
    Returns
    -------
    pd.DataFrame
        从数据库中检索出的指数数据，Index为交易日期。
        如果设置了store为True，则该数据也以key为名保存
        在hdf_file中。
    
    '''
    
    data = retrieve_index_data(index_names, start_date, end_date)
    if store:
        with pd.HDFStore(hdf_file) as file:
            file.put(data_name, data)
    return data

