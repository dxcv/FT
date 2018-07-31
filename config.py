# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-23  10:54
# NAME:FT-config.py

DFCT= r'\\Ft-research\e\Share\Alpha\FYang\factors'
DHTZ= r'\\Ft-research\e\FT_Users\HTZhang'
import os

# DIR_ROOT=r'E:\FT_Users\HTZhang\FT'# review
DIR_ROOT=r'G:\FT_Users\HTZhang\FT'


DRAW=os.path.join(DIR_ROOT,'database','raw')
DCSV=os.path.join(DIR_ROOT,'database','csv')
DPKL=os.path.join(DIR_ROOT,'database','pkl')
DCC=os.path.join(DIR_ROOT,'TMP')
D_FT_RAW=os.path.join(DIR_ROOT,'database','ftresearch_based','raw','pkl')
D_FT_ADJ=os.path.join(DIR_ROOT,'database','ftresearch_based','adjusted','pkl')
D_FILESYNC_RAW= os.path.join(DIR_ROOT,'database','filesync_based','raw')
D_FILESYNC_ADJ= os.path.join(DIR_ROOT,'database','filesync_based','adjusted')
D_DRV=os.path.join(DIR_ROOT,'database','derivatives')

START= '2004-01-01'
END= '2018-01-31'

#single factor
SINGLE_D_RESULT = os.path.join(DIR_ROOT,'singleFactor','result')
SINGLE_D_INDICATOR=os.path.join(DIR_ROOT,'singleFactor','indicators')

'''
all the data saved in SINGLE_D_INDICATOR should be a dataframe with 
'''
SINGLE_D_CHECK=os.path.join(DIR_ROOT,'singleFactor','check')
SINGLE_D_SUMMARY=os.path.join(DIR_ROOT,'singleFactor','summary')
FORWARD_TRADING_DAY=400
FORWARD_LIMIT_Q=400 #400 (trading) days
FORWARD_LIMIT_M=25 #25 (trading) days
LEAST_CROSS_SAMPLE=300

#---------------------------factor combination----------------------------------
DIR_CLEANED=os.path.join(DIR_ROOT,'factor_combination','cleaned')
DIR_BACKTEST=os.path.join(DIR_ROOT,'backtest')
DIR_SIGNAL=os.path.join(DIR_ROOT,'singleFactor','signal')
DIR_SIGNAL_SMOOTHED=os.path.join(DIR_ROOT,'singleFactor','smoothed')
DIR_SINGLE_BACKTEST=os.path.join(DIR_ROOT, 'singleFactor', 'backtest')
DIR_SIGNAL_PARAMETER=os.path.join(DIR_ROOT,'singleFactor','select_parameters')

DIR_SIGNAL_COMB=os.path.join(DIR_ROOT,'singleFactor','combine')
DIR_SIGNAL_SPAN=os.path.join(DIR_ROOT, 'singleFactor', 'signal_spanning')
DIR_HORSE_RACE=os.path.join(DIR_ROOT,'singleFactor','combine','horse_race')
DIR_RESULT_SPAN=os.path.join(DIR_ROOT,'singleFactor','combine','spanning_result')

DIR_BACKTEST_SPANNING=os.path.join(DIR_ROOT,'singleFactor','backtest_spanning')
DIR_MIXED_SIGNAL=os.path.join(DIR_ROOT,'singleFactor','mixed_signal')
DIR_MIXED_SIGNAL_BACKTEST=os.path.join(DIR_ROOT,'singleFactor','mixed_signal_backtest')


# DIR_DM=r'F:\FT_Users\HTZhang\data_mining'
DIR_DM=r'G:\FT_Users\HTZhang\FT\data_mining'
# DIR_DM=os.path.join(DIR_ROOT,'singleFactor','data_mining') #fixme
DIR_DM_TMP=os.path.join(DIR_DM,'tmp')

DIR_DM_RESULT=os.path.join(DIR_DM,'result')
DIR_DM_SIGNAL=os.path.join(DIR_DM,'signal')
DIR_DM_BACKTEST=os.path.join(DIR_DM,'backtest')
DIR_DM_BACKTEST_LONG=os.path.join(DIR_DM,'backtest_long')
DIR_DM_RACE=os.path.join(DIR_DM,'horse_race')


DIR_TMP=os.path.join(DIR_ROOT,'tmp')





# empirical directorys
DIR_EP=r'G:\FT_Users\HTZhang\empirical'
DIR_KOGAN=os.path.join(DIR_EP,'kogan')
