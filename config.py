# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  10:54
# NAME:FT-config.py

DFCT= r'\\Ft-research\e\Share\Alpha\FYang\factors'
DHTZ= r'\\Ft-research\e\FT_Users\HTZhang'
import os

DIR_ROOT=r'E:\FT_Users\HTZhang\FT'# review



# DRAW=r'D:\zht\database\quantDb\internship\FT\database\raw'
DRAW=os.path.join(DIR_ROOT,'database','raw')
# DCSV=r'D:\zht\database\quantDb\internship\FT\database\csv'
DCSV=os.path.join(DIR_ROOT,'database','csv')
# DPKL=r'D:\zht\database\quantDb\internship\FT\database\pkl'
DPKL=os.path.join(DIR_ROOT,'database','pkl')
# DCC=r'D:\zht\database\quantDb\internship\FT\TMP'
DCC=os.path.join(DIR_ROOT,'TMP')
# D_FT_RAW=r'D:\zht\database\quantDb\internship\FT\database\ftresearch_based\raw\pkl'
D_FT_RAW=os.path.join(DIR_ROOT,'database','ftresearch_based','raw','pkl')
# D_FT_ADJ= r'D:\zht\database\quantDb\internship\FT\database\ftresearch_based\adjusted\pkl'
D_FT_ADJ=os.path.join(DIR_ROOT,'database','ftresearch_based','adjusted','pkl')
# D_FILESYNC_RAW= r'D:\zht\database\quantDb\internship\FT\database\filesync_based\raw'
D_FILESYNC_RAW= os.path.join(DIR_ROOT,'database','filesync_based','raw')
# D_FILESYNC_ADJ= r'D:\zht\database\quantDb\internship\FT\database\filesync_based\adjusted'
D_FILESYNC_ADJ= os.path.join(DIR_ROOT,'database','filesync_based','adjusted')
# D_DRV=r'D:\zht\database\quantDb\internship\FT\database\derivatives'
D_DRV=os.path.join(DIR_ROOT,'database','derivatives')

START= '2004-01-01'
END= '2018-01-31'

#single factor
# SINGLE_D_RESULT = r'D:\zht\database\quantDb\internship\FT\singleFactor\result'
SINGLE_D_RESULT = os.path.join(DIR_ROOT,'singleFactor','result')
# SINGLE_D_INDICATOR=r'D:\zht\database\quantDb\internship\FT\singleFactor\indicators'
SINGLE_D_INDICATOR=os.path.join(DIR_ROOT,'singleFactor','indicators')

'''
all the data saved in SINGLE_D_INDICATOR should be a dataframe with 
'''
# SINGLE_D_CHECK=r'D:\zht\database\quantDb\internship\FT\singleFactor\check'
SINGLE_D_CHECK=os.path.join(DIR_ROOT,'singleFactor','check')
# SINGLE_D_SUMMARY=r'D:\zht\database\quantDb\internship\FT\singleFactor\summary'
SINGLE_D_SUMMARY=os.path.join(DIR_ROOT,'singleFactor','summary')
FORWARD_TRADING_DAY=400
FORWARD_LIMIT_Q=400 #400 (trading) days
FORWARD_LIMIT_M=25 #25 (trading) days
LEAST_CROSS_SAMPLE=300

#---------------------------factor combination----------------------------------
# DIR_CLEANED=r'D:\zht\database\quantDb\internship\FT\factor_combination\cleaned'
DIR_CLEANED=os.path.join(DIR_ROOT,'factor_combination','cleaned')
# DIR_BACKTEST=r'D:\zht\database\quantDb\internship\FT\backtest'
DIR_BACKTEST=os.path.join(DIR_ROOT,'backtest')
# DIR_SIGNAL=r'D:\zht\database\quantDb\internship\FT\signal'
DIR_SIGNAL=os.path.join(DIR_ROOT,'signal')
# DIR_BACKTEST_RESULT=r'D:\zht\database\quantDb\internship\FT\backtest\result'
DIR_BACKTEST_RESULT=os.path.join(DIR_ROOT, 'backtest', 'result')



DIR_DM=os.path.join(DIR_ROOT,'singleFactor','data_mining')
DIR_DM_RESULT=os.path.join(DIR_DM,'result')


