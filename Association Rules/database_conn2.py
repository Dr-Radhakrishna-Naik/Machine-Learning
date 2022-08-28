# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:19:52 2022

@author: Dell
"""


pip install psycopg2
import psycopg2
import psycopg2.extras
hostname='127.0.0.1'
database='recruit'
username='postgres'
pwd='root'
port_id=5432
cur=None
conn=None
try:
        conn=psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=pwd,
        port=port_id
        )
except Exception as error:
       print(error)
  

       cur=conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
       cur.execute('DROP TABLE IF EXISTS emp')
       create_script='''CREATE TABLE IF NOT EXISTS emp(
                id  int PRIMARY KEY,
                name varchar(40) NOT NULL,
                salary int,
                dept_id varchar(30))'''
       cur.execute(create_script)
       insert_script='INSERT INTO emp(id,name,salary,dept_id) VALUES(%s,%s,%s,%s)'
      # insert_value=(1,'rk',120000,'cse')
       #insert_value=(2,'pk',100000,'civil')
       #insert_value=(3,'gk',10000,'mech')
       insert_values=[(4,'rk1',12000,'cse'),(5,'rk2',13000,'cse'),(6,'rk3',14000,'cse')]
       for record in insert_values:
           cur.execute(insert_script,record)
       cur.execute('SELECT * FROM emp')
           for record in cur.fetchall():
               print(record['name'],record['salary'])
               #To update the values in database
        
          update_script='UPDATE emp SET salary=(salary+(salary*0.5))'
          cur.execute(update_script)
               
             
       conn.commit()
finally 
        if cur is not None:
            cur.close()
        if  conn is not None:
            conn.close()
