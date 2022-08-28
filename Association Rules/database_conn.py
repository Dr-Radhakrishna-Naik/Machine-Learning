# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:19:52 2022

@author: Dell
"""


pip install psycopg2
import psycopg2
hostname='127.0.0.1'
database='recruit'
username='postgres'
pwd='root'
port_id=5432

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

cur=conn.cursor()

create_script='''CREATE TABLE IF NOT EXISTS emp_new(
                id= int PRIMARY KEY,
                name=varchar(40) NOT NULL,
                salary int,
                dept_id varchar(30))'''
 cur.excecute(create_script)
