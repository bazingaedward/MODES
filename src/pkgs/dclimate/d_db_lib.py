# -*- coding: cp936 -*-
from __future__ import print_function
#-----------------------------------------------------------`-------------------
import time
import sys
import psycopg2


def ins_2_pg(PgStr,sql_list,b_showsql=0):
    print('如果脚本未能运行，可能是pgdb未commit')
    print(u'如果脚本未能运行，可能是pgdb未commit')
    #PgStr="dbname='cfsts' user='cfsts' host='localhost' password='111'"
    try:
        con = psycopg2.connect(PgStr);
    except:
        print('I am unable to connect to the database')
        sys.exit()
    con.autocommit=False

    #print ('%5d,%5.3f sec'%(i,endtime-starttime))
    cursor = con.cursor()
    starttime = time.clock()
    length1 = len(sql_list)
    j=0
    str2=''
    str3=''
    for sql_str in sql_list:
        #print(i,sel_sta[i])
        if(b_showsql):
            print(sql_str)
        cursor.execute(sql_str)
        #------------------------------------
        if(0==j%100 or j==length1-1):
            endtime = time.clock()
            res = endtime-starttime
            precent1 = float(j)*100/float(length1)
            if(0.0==precent1/100.0):
                FullTime=0
                EST=999
            else:
                FullTime= res/(precent1/100.0)
                EST=FullTime-res

            str2='%5.2f%%(%d) cost :%4.1f sec remain:%4.1f sec(Total Elapsed Time%5.1f)'%(precent1,length1,res,EST,FullTime)
            print(str2,end='')
            #str3 = "\b"*len(str2)
            #print(str3,end='')
            print('\r',end='')

            #for j in range(len(str2)):
            #    print("\b",end='')
        #------------------------------------
        j=j+1
        #print(line,type(line))

    con.commit()
    print(str2)
    # Close communication with the database
    cursor.close()
    con.close()

def ins_2_ora(username,password,tsn,sql_list):
    '''
    d_db_lib.ins_2_ora('rdb_insert','ins_2_rdb','172.20.25.5/CD3ZH',AllSqlList)
    '''
    import cx_Oracle
    #con = cx_Oracle.connect("monitor","monitor","orcl")
    #tns = '172.20.25.5/CD3ZH'
    con = cx_Oracle.connect(username,password,tsn)
    cursor = con.cursor()
    ###########################

    #print ('%5d,%5.3f sec'%(i,endtime-starttime))
    length1 = len(sql_list)
    i=0
    str2=''
    str3=''
    starttime = time.clock()
    for sql_str in sql_list:
        #print(i,sel_sta[i])
        cursor.execute(sql_str)
        endtime = time.clock()

        res = endtime-starttime
        precent1 = float(i)*100/float(length1)
        if(0.0==precent1/100.0):
            FullTime=0
            EST=999
        else:
            FullTime= res/(precent1/100.0)
            EST=FullTime-res

        str2='%5.2f%%(%d) cost :%4.1f sec remain:%4.1f sec(%5.1f)'%(precent1,length1,res,EST,FullTime)
        print(str2,end='')
        #str3 = "\b"*len(str2)
        #print(str3,end='')
        print('\r',end='')
        #for j in range(len(str2)):
        #    print("\b",end='')
        i=i+1
    ############################
    print(str2)
    con.commit()
    cursor.close()
    con.close()


def ins_2_fdb(sql_list,username='sysdba',password='masterkey',dbname='10.104.131.33:d:\opt\skyqx.fdb',ptype=0 ):
    import fdb
    con = fdb.connect(dsn=dbname, user=username, password=password)
    length1 = len(sql_list)
    print('sql length = %d'%(length1))
    j=0
    str2=''
    str3=''
    starttime = time.clock()
    for line in sql_list:
        #print(line)
        con.cursor().execute(line)
        #------------------------------------
        if(0==j%100 or j==length1-1 or j==length1):
            endtime = time.clock()
            res = endtime-starttime
            precent1 = float(j)*100/float(length1)
            if(0.0==precent1/100.0):
                FullTime=0
                EST=999
            else:
                FullTime= res/(precent1/100.0)
                EST=FullTime-res

            str2='%5.2f%%(%d) cost :%4.1f sec remain:%4.1f sec(%5.1f)'%(precent1,length1,res,EST,FullTime)
            if(1==ptype):
                print(str2,end='')
                print('\r',end='')
            else:
                print(str2)
            if(j==length1-1 or j==length1):
                print(str2)
                #for j in range(len(str2)):
                #    print("\b",end='')
            #------------------------------------
        j=j+1
        #print(line,type(line))
    con.commit()
    con.close()


def ins_2_firebird(sql_list,username='sysdba',password='masterkey',dbname='10.104.131.33:d:\opt\skyqx.fdb' ):
    import firebirdsql
    con = firebirdsql.connect(dsn=dbname, user=username, \
                          password=password,charset='gb2312')
    con.autocommit=False
    length1 = len(sql_list)
    print('sql length = %d'%(length1))
    j=0
    str2=''
    str3=''
    starttime = time.clock()
    for line in sql_list:
        #print(line)
        con.cursor().execute(line)
        #------------------------------------
        if(0==j%10 or j==length1-1 or j==length1):
            endtime = time.clock()
            res = endtime-starttime
            precent1 = float(j)*100/float(length1)
            if(0.0==precent1/100.0):
                FullTime=0
                EST=999
            else:
                FullTime= res/(precent1/100.0)
                EST=FullTime-res

            str2='%5.2f%%(%d) cost :%4.1f sec remain:%4.1f sec(%5.1f)'%(precent1,length1,res,EST,FullTime)
            print(str2,end='')
            print('\r',end='')
            if(j==length1-1 or j==length1):
                print(str2)

            #for j in range(len(str2)):
            #    print("\b",end='')
        #------------------------------------
        j=j+1
        #print(line,type(line))
    con.commit()
    con.close()

######################################################
#导出观测数据历史数据日值
######################################################
def exp_history_days_obs(s_year1,s_year2,str_date1,day_count,i_sta_type,ival_type=1):
    '''
##################################################
导出观测数据历史数据
s_year1 开始年
s_year2 结束年
s_mon 月份
count 月份的个数为1时是月，
i_sta_type 站点类型
    '''
    FileName1 = 'obs_days_%s_%s_%s_%d_%d_%d.txt'%(s_year1,s_year2,str_date1,day_count,i_sta_type,ival_type)
    import os
    if(os.path.isfile(FileName1)):
        return FileName1

    print(s_year1,s_year2,str_date1)
    import psycopg2
    PgStr="host='10.104.195.82' user='skyqx' password='111' dbname='skyqx' "
    con = psycopg2.connect(PgStr);
    cursor = con.cursor()
    #cursor.execute('select * from "EXP03_ZINDEX_YEAR_MON2MON"(%s,%s,%s,%i)'%(s_year1,s_year2,s_mon,count))
    #select * from day."EXP01_STA_HIS_DAY2DAY_OBS"(2009,2013,'2013-09-01',30,45,1)
    str1 = 'select * from day."EXP01_STA_HIS_DAY2DAY_OBS"(%s,%s,\'%s\',%d,%d,%d)'%(s_year1,s_year2,str_date1,day_count,i_sta_type,ival_type)
    print(str1)
    cursor.execute(str1)

    sql_result=cursor.fetchall()
    print('OUTPUT ROWS=',len(sql_result))
    list1 = []
    for i in range(len(sql_result)):
        #print(sql_result[i][0])
        list1.append(sql_result[i][0])


    file1 = open(FileName1,'w')
    file1.writelines(["%s\n" % item  for item in list1])
    file1.close()

    cursor.close()
    con.close()
    return FileName1

######################################################
#导出观测数据历史数据
######################################################
def exp_history_months_obs(s_year1,s_year2,s_mon,count,i_sta_type,obs_type=1):
    '''
##################################################
导出观测数据历史数据
s_year1 开始年
s_year2 结束年
s_mon 月份
count 月份的个数为1时是月，
i_sta_type 站点类型
    '''
    print(s_year1,s_year2,s_mon)
    import os

    FileName1 = 'obs_months_%s_%s_%s_%d_%d_%d.txt'%(s_year1,s_year2,s_mon,count,i_sta_type,obs_type)
    FileName1 = os.path.join(r'.\tmp',FileName1)
    if(os.path.isfile(FileName1)):
        return FileName1

    import psycopg2
    PgStr="host='10.104.195.82' user='skyqx' password='111' dbname='skyqx' "
    con = psycopg2.connect(PgStr);
    cursor = con.cursor()
    #cursor.execute('select * from "EXP03_ZINDEX_YEAR_MON2MON"(%s,%s,%s,%i)'%(s_year1,s_year2,s_mon,count))
    cursor.execute('select * from "EXP01_STA_HIS_MON_OBS"(%s,%s,%s,%d,%d,%d)'%(s_year1,s_year2,s_mon,count,i_sta_type,obs_type))

    sql_result=cursor.fetchall()
    print(len(sql_result))
    list1 = []
    for i in range(len(sql_result)):
        #print(sql_result[i][0])
        list1.append(sql_result[i][0])


    file1 = open(FileName1,'w')
    file1.writelines(["%s\n" % item  for item in list1])
    file1.close()

    cursor.close()
    con.close()
    return FileName1

######################################################
#导出环流指数历史数据
#2015-02-15
######################################################
def exp_idx(s_year1,s_year2,s_mon,count,interval=1):
    '''
######################################################
#导出环流指数历史数据
######################################################
    '''
    FileName1 = 'idx_%s_%s_%s_%d_%d.txt'%(s_year1,s_year2,s_mon,count,interval)
    import os
    if(os.path.isfile(FileName1)):
        return FileName1

    print(s_year1,s_year2,s_mon)
    import psycopg2
    PgStr="host='10.104.195.82' user='skyqx' password='111' dbname='skyqx' "
    con = psycopg2.connect(PgStr);
    cursor = con.cursor()

    str1 = 'select * from "EXP03_ZINDEX_YEAR_MON2MON"(%s,%s,%s,%d,%d)'%(s_year1,s_year2,s_mon,count,interval)
    print('SQL Str=',str1)
    cursor.execute(str1)
    sql_result=cursor.fetchall()
    print('Index OUTPUT ROWS=',len(sql_result))
    list1 = []
    for i in range(len(sql_result)):
        #print(sql_result[i][0])
        list1.append(sql_result[i][0])


    file1 = open(FileName1,'w')
    file1.writelines(["%s\n" % item  for item in list1])
    file1.close()

    cursor.close()
    con.close()
    return FileName1


######################################################
#导出观测数据历史数据日值
######################################################
def exp_month_count_obs(i_year,i_mon,mon_count,i_sta_type,ival_type=1,i_usefile=1):
    '''
##################################################
导出观测数据历史数据
i_year 开始年月
i_mon 月份数
mon_count
i_sta_type 站点类型
    '''
    import os
    FileName1 = 'obs_mon_count_%d_%d_%d_%d_%d.txt'%(i_year,i_mon,mon_count,i_sta_type,ival_type)

    if(not os.path.isdir(r'.\tmp')):
        os.mkdir('tmp')


    FileName1 = os.path.join(r'.\tmp',FileName1)

    if(os.path.isfile(FileName1) and i_usefile):
        return FileName1

    print(i_year,i_mon,mon_count)
    import psycopg2
    PgStr="host='10.104.195.82' user='skyqx' password='111' dbname='skyqx' "
    con = psycopg2.connect(PgStr);
    cursor = con.cursor()
    #cursor.execute('select * from "EXP03_ZINDEX_YEAR_MON2MON"(%s,%s,%s,%i)'%(s_year1,s_year2,s_mon,count))
    #select * from day."EXP01_STA_HIS_DAY2DAY_OBS"(2009,2013,'2013-09-01',30,45,1)
    s_date1 = '%04d-%02d-01'%(i_year,i_mon)
    str1 = 'select * from "EXP02_STA_MON_OBS"(\'%s\',%d,%d,%d) where n_value2 is not NULL'%(s_date1,mon_count,i_sta_type,ival_type)
    print(str1)
    cursor.execute(str1)

    sql_result=cursor.fetchall()

    #################################################
    row_type_list=[]
    for i in range(len( cursor.description ) ):
        row=cursor.description[i]
        #print(row)
        row_type_list.append(row[1])
        #################################################
    #sys.exit(0)

    print(len(sql_result))
    list1 = []
    for i in range(len(sql_result)):
        #print(sql_result[i][0])
        print(sql_result[i])
        str1 = ''
        for jj in range(len(row_type_list)):

            if( 23==row_type_list[jj] ):
                if(''==str1):
                    str1 = str1+' %d'%sql_result[i][jj]
                else:
                    str1 = str1+' %d'%sql_result[i][jj]

            if( 1700==row_type_list[jj] ):
                if(''==str1):
                    str1 = str1+' %8.2f'%sql_result[i][jj]
                else:
                    str1 = str1+' %8.2f'%sql_result[i][jj]

            if( 1042==row_type_list[jj] ):
                if(''==str1):
                    str1 = str1+' %s'%sql_result[i][jj]
                else:
                    str1 = str1+' %s'%sql_result[i][jj]

                #list2.append(line1)
            #list1.append(sql_result[i][0])
        #print(' '.join(str(list2)))
        #list1.append( ' '.join(str(list2)) )
        list1.append(str1)

    file1 = open(FileName1,'w')
    file1.writelines(["%s\n" % item  for item in list1])
    file1.close()

    cursor.close()
    con.close()
    return FileName1


######################################################
#导出观测数据历史数据日值
######################################################
def exp_day_count_obs(i_sta_type,date_str1,date_str2,ival_type=1,i_usefile=1):
    '''
##################################################
导出观测数据历史数据
i_year 开始年月
i_mon 月份数
mon_count
i_sta_type 站点类型
i_usefile 是否利用已经生成的数据文件
    '''
    import os
    if(not os.path.isdir('tmp')):
        os.mkdir("tmp")

    FileName1 = 'obs_day_count_%d_%s_%s_%d.txt'%(i_sta_type,date_str1,date_str2,ival_type)
    FileName1 = os.path.join('./tmp',FileName1)


    if(os.path.isfile(FileName1) and i_usefile):
        return FileName1

    print(i_sta_type,date_str1,date_str2,ival_type)
    import psycopg2
    PgStr="host='10.104.195.82' user='skyqx' password='111' dbname='skyqx' "
    con = psycopg2.connect(PgStr);
    cursor = con.cursor()
    #cursor.execute('select * from "EXP03_ZINDEX_YEAR_MON2MON"(%s,%s,%s,%i)'%(s_year1,s_year2,s_mon,count))
    #select * from day."EXP01_STA_HIS_DAY2DAY_OBS"(2009,2013,'2013-09-01',30,45,1)

    str1 = 'select * from day."EXP02_STA_DAY2DAY_OBS"(%d,\'%s\',\'%s\',%d)'%(i_sta_type,date_str1,date_str2,ival_type)
    print(str1)
    cursor.execute(str1)

    sql_result=cursor.fetchall()

    #################################################
    row_type_list=[]
    for i in range(len( cursor.description ) ):
        row=cursor.description[i]
        #print(row)
        row_type_list.append(row[1])
        #################################################
    #sys.exit(0)

    print('OUTPUT ROWS=',len(sql_result))
    list1 = []
    for i in range(len(sql_result)):
        #print(sql_result[i][0])
        str1 = ''
        for jj in range(len(row_type_list)):

            #print( row_type_list[jj] )

            if( 23==row_type_list[jj] ):
                if(''==str1):
                    str1 = str1+' %d'%sql_result[i][jj]
                else:
                    str1 = str1+' %d'%sql_result[i][jj]

            if( 1700==row_type_list[jj] ):
                if(''==str1):
                    str1 = str1+' %8.2f'%sql_result[i][jj]
                else:
                    str1 = str1+' %8.2f'%sql_result[i][jj]

            if( 1042==row_type_list[jj] or 1043==row_type_list[jj] ):  #1042 char 1043 varchar
                if(''==str1):
                    str1 = str1+' %s'%sql_result[i][jj]
                else:
                    str1 = str1+' %s'%sql_result[i][jj]

                    #list2.append(line1)
                    #list1.append(sql_result[i][0])
            #print(' '.join(str(list2)))
        #list1.append( ' '.join(str(list2)) )
        list1.append(str1)

    file1 = open(FileName1,'w')
    file1.writelines(["%s\n" % item  for item in list1])
    file1.close()

    cursor.close()
    con.close()
    return FileName1

#con = cx_Oracle.connect("monitor","monitor","orcl")
def from_fb_sql2insert(PgTableName,sql_str = 'select * from  sky_station'):
    import firebirdsql,string

    #and rownum<5
    #con = cx_Oracle.connect("cipas","cipas","orcl")
    con = firebirdsql.connect(dsn=r'10.104.131.33:D:\opt\SKYQX.FDB', user='sysdba',password='masterkey',charset='gb2312')
    # Create a Cursor object that operates in the context of Connection con:
    #cur = con.cursor()
    #print(con.dsn)
    #print(con.version)
    cursor = con.cursor()
    #sql_str = 'select * from tab'u
    cursor.execute(sql_str)
    #con.commit()
    sql_result=cursor.fetchall()
    ####################################################
    fields=[]
    for row in cursor.description:
        fields.append(row[0])
        #print(row)
    varstr = string.join(fields,',');
    ##print(varstr)
    #####################################################
    ##print(type(sql_result))
    ##print(sql_result[0])
    #line1 = sql_result[0]
    #print(line1,type(line1))
    #####################################################
    cols = len(cursor.description)
    row_type_list=[]
    for i in range(cols):
        row=cursor.description[i]
        print(row)
        row_type_list.append(row[1])
        #####################################################
    AllSqlList=[]
    starttime = time.clock()
    length1 = len(sql_result)
    j=0
    str2=''
    str3=''
    for line1 in sql_result:
        #------------------------------------
        if(0==j%1000 or j==length1-2):
            #print(j)
            endtime = time.clock()
            res = endtime-starttime
            precent1 = float(j)*100/float(length1)
            if(0.0==precent1/100.0):
                FullTime=0
                EST=999
            else:
                FullTime= res/(precent1/100.0)
                EST=FullTime-res

            str2='%5.2f%%(%d) cost :%4.1f sec remain:%4.1f sec(%5.1f)'%(precent1,length1,res,EST,FullTime)
            print(str2,end='')
            str3 = "\b"*len(str2)
            print(str3,end='')

            #for j in range(len(str2)):
            #    print("\b",end='')
        #print(line,type(line))
        #------------------------------------
        j=j+1

        Field=[]
        for i in range(cols):
            #row=cursor.description[i]
            #print(row_type_list[i])
            #continue
            if(None == line1[i]):
                Field.append('NULL')
                continue

            if(row_type_list[i]==496):
                Field.append('%d'%line1[i])
                continue

            if(row_type_list[i]==448):
                str1 = line1[i]
                str1 = str1.decode('gb2312')
                #Field.append("'"+line1[i]+"'")
                Field.append("'"+str1+"'")
                continue



            #if(row_type_list[i]==cx_Oracle.DATETIME):
            #    Field.append("'"+datetime.datetime.strftime(line1[i],'%Y-%m-%d')+"'")
            #    continue
                #print(line1[i],datetime.datetime.strftime(line1[i],'%Y-%m-%d'))


                #print(line1[i],row,type(line1[i]))
                #else:
                #    print(line1[i],row[1])
                #sys.exit(0)
        #sys.exit(0)
        varstr2 = string.join(Field,',');
        #print(len(line1),len(Field))
        #print(varstr2)
        sqlstr='insert into %s(%s) values(%s)'%(PgTableName,varstr,varstr2)
        #print(sqlstr)
        AllSqlList.append(sqlstr)
        #break
        #for line in sql_result:
        #    print(line,type(line))
    print(str2)
    cursor.close()
    con.close()
    return AllSqlList
