# coding:utf-8

import urllib
from bs4 import BeautifulSoup
import datetime as dt
import os

FILE_STORE_DIR = '/Users/qiukaixiang/Desktop'

html = urllib.urlopen('http://www.ifeng.com/')
soup = BeautifulSoup(html.read(),'html.parser')
links = soup.find('div',id='headLineDefault').find_all('a')

## open file to save data
now = dt.datetime.now()
with open(os.path.join(FILE_STORE_DIR,'web.txt'),'w') as fh:
    for link in links:
        fh.write('{},{}\n'.format(link.string.encode('utf-8'),link['href'].encode('utf-8')))

    fh.close()


