# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:45:56 2022

@author: Dell
"""

from flask import Flask
#wsgi application
app=Flask(__name__)

@app.route('/')
def results():
    return "Results of first sem examination"
@app.route('/success/<int:score>')
def success(score):
    return "The person has passed with"+str(score)
@app.route('/fail/<int:score>')
def fail(score):
    return "The person has failed with"+str(score)

if __name__=='__main__':
    app.run(debug=True)