#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse


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