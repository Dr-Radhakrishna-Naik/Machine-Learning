# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:12:46 2022

@author: Dell
"""

from flask import Flask, render_template
app = Flask(__name__)
@app.route('/hello/<user>')
def hello_name(user):
    return render_template('hello.html', name=user)
if __name__ == '__main__':
    app.run(debug=True)