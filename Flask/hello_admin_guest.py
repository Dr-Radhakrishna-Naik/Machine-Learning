# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 19:07:29 2022

@author: Dell
"""

from flask import Flask,redirect,url_for
app=Flask(__name__)
@app.route('/admin')
def admin():
    return 'Hello admin'

@app.route('/guest/<guest>')
def hello_guest(guest):
     return 'hello %s as Guest'%guest
@app.route('/user/<name>')
def hello_user(name):
    if name=='admin':
        return redirect(url_for('admin'))
    else:
        return redirect(url_for('hello_guest',guest=name))    
if __name__=='__main__':
    app.run(debug=True)