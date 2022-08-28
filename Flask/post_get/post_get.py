# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:18:02 2022

@author: Dell
"""

from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/sales_a')
def sales_a():
   return render_template('index1.html')
@app.route('/result1',methods=['POST', 'GET'])
def result1():
    if request.method=='POST':
       result1=request.form
       return render_template("table1.html",result1=result1)
@app.route('/sales_b')
def sales_b():
   return render_template('index2.html')
@app.route('/result2',methods=['POST', 'GET'])
def result2():
       if request.method=='POST':
          result2=request.form
          return render_template("table2.html",result2=result2)
@app.route('/sales_c')
def sales_c():
   return render_template('index3.html')
@app.route('/result3',methods=['POST', 'GET'])
def result3():
       if request.method=='POST':
          result3=request.form
          return render_template("table3.html",result3=result3)

@app.route('/sales_d')
def sales_d():
   return render_template('index4.html')
@app.route('/result4',methods=['POST', 'GET'])
def result4():
       if request.method=='POST':
          result4=request.form
          return render_template("table4.html",result4=result4)
@app.route('/sales_e')
def sales_e():
   return render_template('index5.html')
@app.route('/result5',methods=['POST', 'GET'])
def result5():
       if request.method=='POST':
          result5=request.form
          return render_template("table5.html",result5=result5)
@app.route('/sales_f')
def sales_f():
   return render_template('index6.html')
@app.route('/result6',methods=['POST', 'GET'])
def result6():
       if request.method=='POST':
          result6=request.form
          return render_template("table6.html",result6=result6)
@app.route('/sales_g')
def sales_g():
   return render_template('index7.html')
@app.route('/result7',methods=['POST', 'GET'])
def result7():
       if request.method=='POST':
          result7=request.form
          return render_template("table7.html",result7=result7)

@app.route('/sales_h')
def sales_h():
   return render_template('index8.html')
@app.route('/result8',methods=['POST', 'GET'])
def result8():
       if request.method=='POST':
          result8=request.form
          return render_template("table8.html",result8=result8)








if __name__ == '__main__':
    app.run(debug=True)