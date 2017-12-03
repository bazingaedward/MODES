# -*- coding: cp936 -*-
'''
2012-02-24
杜良敏专业科学计算，绘图，插值，预报专用程序库
后期不断完善更新
'''
import sys, os, math

__version__ = '0.98'
__author__ = 'Longman Du'
Spatial_Data = os.sep.join([os.path.dirname(__file__), 'spatialdat'])
Station_Data = os.sep.join([os.path.dirname(__file__), 'stationdat'])
Level_Path = os.sep.join([os.path.dirname(__file__), 'lev'])
Tmp_Path = os.sep.join([os.path.dirname(__file__), 'tmp'])
Magick_Convert = os.sep.join([os.path.dirname(__file__), 'bin\convert.exe'])
#print(Spatial_Data)