from email.policy import default
import re
from flask import Flask, render_template,request, url_for, redirect
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.svm import SVR 
svr_rbf= SVR(kernel='rbf',C=1e3, gamma=0.00001)
# from google.colab import drive
# drive.mount('/content/drive')
import matplotlib.pyplot as plt

df= pd.read_csv('bitcoin.csv')
date= df['Date']
def days(d1,d2):
  date_format = "%d-%m-%Y"
  a = datetime.strptime(d1, date_format)
  b = datetime.strptime(d2, date_format)
  delta = b - a
  return delta.days


app = Flask(__name__)

    
@app.route('/')
def entry_point():
    return render_template('index.html')

@app.route('/open',methods=['POST','GET'])
def predict_open():
    X=[str(x) for x in request.form.values()]
    # x= str(request.form)
    print(X)
    d1= 	'15-04-2022'
    d2= X[0]
    num= days(d1,d2)
    if(num>0):
        print(type(df['Open'][0]))
        df['open_pred']= df[['Open']].shift(num)
        df1= df.dropna()
        x= df1['Open'].to_numpy().reshape(-1, 1)
        X= df1['Open'].to_numpy()
        Y= df1['open_pred'].to_numpy()
        values= df1['open_pred'].to_numpy()
        dates= df1['Date']
        if num<=100:
            x= x[2000:]
            Y= Y[2000:]
        svr_rbf.fit(x,Y)
        data=[['Date','price']]
        for i in range(len(dates)):
            data.append([df['Date'][i],X[i]])
        return render_template('open.html', pred=str(svr_rbf.predict(np.array(39978.73438).reshape(-1, 1))[0]),labels=X[2500:],values= values,data=data)
    else:
        return render_template('open.html', pred=str(df['Open'][-num]),labels=X[2500:],values= values, dates=dates,data=data)
    

@app.route('/close',methods=['POST','GET'])
def predict_close():
    X=[str(x) for x in request.form.values()]
    # x= str(request.form)
    print(X)
    d1= 	'15-04-2022'
    d2= X[0]
    num= days(d1,d2)
    if(num>0):
        print(type(df['Close'][0]))
        df['close_pred']= df[['Close']].shift(num)
        df1= df.dropna()
        x= df1['Close'].to_numpy().reshape(-1, 1)
        X= df1['Close'].to_numpy()
        Y= df1['close_pred'].to_numpy()
        values= df1['close_pred'].to_numpy()
        dates= df1['Date']
        if num<=100:
            x= x[2000:]
            Y= Y[2000:]
        svr_rbf.fit(x,Y)
        data=[['Date','price']]
        for i in range(len(dates)):
            data.append([df['Date'][i],X[i]])
        return render_template('close.html', pred=str(svr_rbf.predict(np.array(40121.46875).reshape(-1, 1))[0]),labels=X[2500:],values= values,data=data)
    else:
        return render_template('close.html', pred=str(df['Close'][-num]),labels=X[2500:],values= values, dates=dates,data=data)


@app.route('/volume',methods=['POST','GET'])
def predict_volume():
    X=[str(x) for x in request.form.values()]
    # x= str(request.form)
    print(X)
    d1= 	'15-04-2022'
    d2= X[0]
    num= days(d1,d2)
    if(num>0):
        print(type(df['Volume'][0]))
        df['volume_pred']= df[['Volume']].shift(num)
        df1= df.dropna()
        x= df1['Volume'].to_numpy().reshape(-1, 1)
        X= df1['Volume'].to_numpy()
        Y= df1['volume_pred'].to_numpy()
        values= df1['volume_pred'].to_numpy()
        dates= df1['Date']
        if num<=100:
            x= x[2000:]
            Y= Y[2000:]
        svr_rbf.fit(x,Y)
        data=[['Date','price']]
        for i in range(len(dates)):
            data.append([df['Date'][i],X[i]])
        return render_template('volume.html', pred=str(svr_rbf.predict(np.array(25157980160).reshape(-1, 1))[0]),labels=X[2500:],values= values,data=data)
    else:
        return render_template('volume.html', pred=str(df['Volume'][-num]),labels=X[2500:],values= values, dates=dates,data=data)


if __name__ == '__main__':
    app.run(debug=True)