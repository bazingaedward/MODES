#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import ConfigParser
import argparse

from HTMLParser import HTMLParser
import urllib
import subprocess
from glob import glob

class Settings(object):
    """配置参数类"""

    # 是否使用执行程序目录下的配置文件,否则使用当前路径下的配置文件
    __use_execute_path = False

    # 配置文件名
    __config_file = None

    __config = None

    def __init__(self):

        if self.__use_execute_path:
            path = os.path.dirname(sys.argv[0])
            path = os.path.abspath(path)
        else:
            path = os.getcwd()

        self.__config_file = os.path.join(path, 'settings.ini')

        self.__config = ConfigParser.ConfigParser()
        self.__config.read(self.__config_file)

    def getcfg(self, section, option, default=None, type=str):
        """返回配置参数项"""

        ret = default
        getfn = self.__config.get

        if type == int:
            getfn = self.__config.getint
        elif type == float:
            getfn = self.__config.getfloat
        elif type == bool:
            getfn = self.__config.getboolean

        try:
            ret = getfn(section, option)
        except Exception, e:
            print(u'**警告** 从{}中读取配置[{}]失败! 原因:{}'
                  .format(self.__config_file, '/'
                          .join([section, option]), e.message))

        return ret

    def setcfg(self, section, option, value):
        """设置并保存配置参数项"""

        try:
            self.__config.set(section, option, value)
            self.__config.write(open(self.__config_file, 'wb'))
        except Exception, e:
            print(u'**警告** 向{}中写入配置[{}]失败! 原因:{}'
                  .format(self.__config_file, '/'
                          .join([section, option]), e.message))

class Argument(object):
    """命令行解析类"""

    command_args = None
    """命令行参数"""

    __parser = None

    __args = None

    def __init__(self):
        self.__parser = argparse.ArgumentParser()

        if self.command_args is not None:
            for args in self.command_args:
                self.__parser.add_argument(*args[0], **args[1])

        self.__args = self.__parser.parse_args()

    @property
    def args(self):
        return self.__args

    def parse_args(self):
        pass

class parser(HTMLParser):
    """parser类：过滤网页中的POMMA数据文件名"""

    def __init__(self):
        HTMLParser.__init__(self)
        self.urls = []  # 将获取到的POMMA数据名称保存在urls中

    def handle_starttag(self, tag, attrs):
        pass

    def handle_data(self, data):
        # 设置过滤条件，POMMA的数据开头三个字母为mac
        if data[0:3] == 'mac':
            self.urls.append(data)

    def handle_endtag(self, tag):
        pass


class POMMA(Argument, Settings):
    """POMMA数据下载"""
    M24_TYPE = ['m24a','m24b','m24c']
    # M24_TYPE = ['m24a']
    M24_CATALOG = ['emn','e10','e09','e08','e07','e06','e05','e04','e03','e02','e01','e00']
    # M24_CATALOG = ['emn']
    parameters = dict()

    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('path',), {'help': u'输出的文件根目录路径', 'type': str, 'nargs': 1}],
        # [('-t','--type',), {'help': u'选择poama下载模式[all/latest],默认latest,', 'choices': ['all', 'latest'], 'nargs': 1}],
        [('-v','--verbose',), {'help': u'输出详细信息', 'action':"store_true"}]
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)
        self.parse_args()

    def parse_args(self):
        """ 解析参数 """
        ## parameter: path
        if self.args.path:
            self.parameters['path'] = self.args.path[0]
        else:
            print('{} not found!'.format(self.args.path[0]))
            exit(-1)

        ## parameter: verbose
        if self.args.verbose:
            print("============parameters===========")
            for key in self.parameters:
                print(key,'= ',self.parameters[key])

        self.process()

    def process(self):
        """ 处理 """

        """ 下载文件到指定位置 """
        for type1 in self.M24_TYPE:
            for type2 in self.M24_CATALOG:
                storePath = os.path.abspath(os.path.join(self.parameters['path'],type1,type2))
                ## check file path
                if not os.path.exists(storePath):
                    os.makedirs(storePath)
                    if self.args.verbose:
                        print('[', type1, ']','[', type2, ']','create directory ',storePath)

                url = 'http://opendap.bom.gov.au:8080/thredds/catalog/poama/realtime/monthly/{}/{}/catalog.html'.\
                        format(type1,type2)

                newNames = self.getFileNames(url)
                oldNames = [os.path.basename(one) for one in glob(os.path.join(storePath,'mac_*.nc'))]

                def downloadIt(filename):
                    """ 下载指定文件名的数据 """

                    ## dirPath:url的上层地址
                    dirPath = os.path.dirname(url)

                    ## dataURL:最终的POMMA下载根目录（POMMA存放文件路径与tomcat前台显示路径不同）
                    dataURL = dirPath.replace('catalog', 'fileServer')

                    ## download data with wget command
                    p = subprocess.Popen(['wget', os.path.join(dataURL,filename),'-P',storePath],\
                                         stdout=subprocess.PIPE)
                    p.communicate()

                if newNames:
                    for item in newNames:
                        if item not in oldNames:
                            if self.args.verbose:
                                print('[', type1, ']','[', type2, ']','downloading ',item)
                            downloadIt(item)
                else:
                    print(u'该URL下未找到相关文件\n',url)
                    exit(-1)






    def getFileNames(self,url):
        """ 将poama数据链接获取下来,并转变成可下载路径,最后保存到本地文件. """

        ## 解析url
        self.parser = parser()
        self.parser.feed(urllib.urlopen(url).read())
        self.parser.close()

        ## 返回的文件名已经排好序 eg. mac_20161117.nc
        return self.parser.urls




if __name__ == '__main__':
    temp = POMMA()
