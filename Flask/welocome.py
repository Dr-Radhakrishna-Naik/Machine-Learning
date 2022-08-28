# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:32:20 2022

@author: Dell
"""
from flask import Flask
app=Flask(__name__)

@app.route('/')
def welcome():
     return "welcome"
 
if __name__=='__main__':
    app.run(debug=True)