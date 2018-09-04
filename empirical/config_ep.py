# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-31  21:34
# NAME:FT_hp-config.py
import os

# empirical directorys
DIR_EP=r'G:\FT_Users\HTZhang\empirical'
DIR_BASEDATA=os.path.join(DIR_EP,'basedata')


#------------------kogan-----------------------------------------
DIR_KOGAN=os.path.join(DIR_EP,'kogan')
DIR_KOGAN_RESULT=os.path.join(DIR_EP,'kogan','results')
DIR_B1=os.path.join(DIR_KOGAN,'bootstrap1')

NUM_FACTOR=3
CRITICAL=0.05

#-----------------yan--------------------------------
DIR_YAN=os.path.join(DIR_EP,'yan')


#------------------chordia------------------------------
DIR_CHORDIA=os.path.join(DIR_EP,'chordia')



#--------------------data-mining------------------------
DIR_DM=os.path.join(DIR_EP,'data_mining')
DIR_DM_INDICATOR=os.path.join(DIR_DM,'indicator')
DIR_DM_NORMALIZED=os.path.join(DIR_DM,'normalized')
PERIOD_THRESH=60

