# -*- coding: cp936 -*-
from dfunc import mytic,mytoc
#
import os,sys
#
import ConfigParser
#
#import fdb
#

#print(getDClimDiagini())

def getMode1Path():
    MODES1_PATH = os.environ.get('MODES1')
    return MODES1_PATH


def getDClimDiagini(MODES_INI_FILE='MODES1.ini'):

    #def optionxform(self, optionstr):
    #    #return optionstr.lower()
    #    return optionstr

    #config1 = ConfigParser.ConfigParser()
    #config1.readfp(open(FileName))
    global MODES_PATH
    MODES1_PATH = os.environ.get('MODES1')

    config1 = ConfigParser.ConfigParser()
    config1.optionxform = str
    MODES_INI_FILE = os.path.join(MODES1_PATH,MODES_INI_FILE)
    config1.readfp(open(MODES_INI_FILE))

    dict1 = {}


    #93##########MUdAY_PATH################
    str1 = config1.get("GLOBAL","MUMON_PATH")
    str1 = str1.replace('${INITDIR}',MODES1_PATH)
    dict1['MUMON_PATH']=str1

    ###########CIPAS MUDAY###############
    str1 = config1.get("GLOBAL","CIPAS_FTP_HOST")
    dict1['CIPAS_FTP_HOST']=str1

    str1 = config1.get("GLOBAL","CIPAS_FTP_USERNAME")
    dict1['CIPAS_FTP_USERNAME']=str1

    str1 = config1.get("GLOBAL","CIPAS_FTP_PASSWORD")
    dict1['CIPAS_FTP_PASSWORD']=str1

    str1 = config1.get("GLOBAL","CIPAS_MUMOM_PATH")
    dict1['CIPAS_MUMOM_PATH']=str1


    sections1 = config1.sections()
    print 'section:', sections1

    options1 = config1.options("GLOBAL")
    print 'options:', options1
    dict1.clear()

    for line1 in options1:
        str1 = config1.get("GLOBAL",line1)
        #print(type(str1))
        dict1[line1]=str1.replace('${INITDIR}',MODES1_PATH)

    #dict2 = config1.items('GLOBAL')
    #print(type(dict2[0]))

    #print(dict1['fbdbfile'])
    #print(dict1)

    return dict1

if __name__ == "__main__" :
    getDClimDiagini()