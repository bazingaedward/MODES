#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from HTMLParser import HTMLParser
import urllib
from command import Argument, Settings

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
    storePath = '.'  # 默认解析文件保存在当前目录

    # 命令行参数,参考 https://docs.python.org/2/howto/argparse.html
    command_args = [
        [('url',), {'help': u'文件目录的URL地址', 'type': str, 'nargs': 1}],
        [('-o',), {'help': u'输出的文件路径', 'type': str, 'nargs': 1}],
    ]

    def __init__(self):
        Argument.__init__(self)
        Settings.__init__(self)

        self.parse_args()

    def parse_args(self):
        url = self.args.url[0] if self.args.url[0] is not None else ''
        if self.args.o:
            self.storePath = self.args.o[0]

        # 解析url
        self.parser = parser()
        self.parser.feed(urllib.urlopen(url).read())

        # 判断结果
        if self.parser.urls:
            # dirPath:url的上层地址
            dirPath = os.path.dirname(url)
            # dataURL:最终的POMMA下载根目录（POMMA存放文件路径与tomcat前台显示路径不同）
            dataURL = dirPath.replace('catalog', 'fileServer')

            outfile = os.path.join(self.storePath, 'catalog.txt')
            with open(outfile, 'w') as f:
                for item in self.parser.urls:
                    f.write(os.path.join(dataURL, item) + '\n')

            self.parser.close()
            f.close()
        else:
            print(u'该url下未找到数据。')


if __name__ == '__main__':
    temp = POMMA()
