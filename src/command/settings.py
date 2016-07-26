#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import ConfigParser


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
