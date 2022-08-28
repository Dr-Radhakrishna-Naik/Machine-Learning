# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:12:53 2022

@author: Dell
"""

from flask import Flask
#wsgi app
app=Flask(__name__)
@app.route('/')
def welcome():
    return "Welcome"
@app.route('/members')
def members():
    return "hi welcome"
if __name__=='__main__':
    app.run(debug=True)