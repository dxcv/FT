# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-06-04  00:25
# NAME:FT-get_all_youtube_urls.py

from urllib.request import urlopen#用于获取网页
from bs4 import BeautifulSoup#用于解析网页

html=open(r'e:\a\html.html',encoding='utf8').read()
bsObj = BeautifulSoup(html, 'html.parser')
t1 = bsObj.find_all('a')

urls=[]
for t2 in t1:
    t3 = t2.get('href')
    if t3:
        urls.append(t3)
        print(t3)

urls=sorted(list(set(urls)))

with open(r'e:\a\urls.txt','w') as f:
    f.write('\n'.join(urls))

