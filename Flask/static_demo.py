# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 19:10:17 2022

@author: Dell
"""

from flask import Flask, render_template
app =Flask(__name__)
@app.route("/")
def index():
   return render_template("index.html")
if __name__ == '__main__':
   app.run(debug=True)