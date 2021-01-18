import sys
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import numpy as np
import sklearn.metrics as sm
from datetime import datetime
def transData(temp):
    temp=temp.reset_index(drop=True)
    for i in range(0,len(temp['Date'])):
        time1 = datetime.strptime(temp['Date'][i], "%d/%m/%Y")
        temp['Date'][i]=time1.strftime('%Y-%m-%d')
    return temp
lupa_data=pd.read_csv("../datasets/Water_Spring_Lupa.csv", encoding='utf-8', sep=',')
lupa_data.dropna(axis=0, how='any', inplace=True)
lupa_data=transData(lupa_data)
lupa_data.to_csv('lupa_data.csv',sep=',', header=True, index=True)
madonna_data=pd.read_csv("../datasets/Water_Spring_Madonna_di_Canneto.csv", encoding='utf-8', sep=',')
madonna_data.dropna(axis=0, how='any', inplace=True)
madonna_data=transData(madonna_data)
madonna_data.to_csv('madonna_data.csv',sep=',', header=True, index=True)
amiata_data=pd.read_csv("../datasets/Water_Spring_Amiata.csv", encoding='utf-8', sep=',')
amiata_data.dropna(axis=0, how='any', inplace=True)
amiata_data=transData(amiata_data)
amiata_data.to_csv('amiata_data.csv',sep=',', header=True, index=True)
print(madonna_data.columns)
result=madonna_data['Flow_Rate_Madonna_di_Canneto']
madonna_data.drop(['Flow_Rate_Madonna_di_Canneto','Date'], axis=1, inplace=True)
divideRatio=0.8
dividingInt = int(madonna_data.shape[0]*divideRatio)
train_data = madonna_data[:dividingInt]
test_data = madonna_data[dividingInt:]
train_y=result[:dividingInt]
test_y=result[dividingInt:]
scoreList = []
alphas = 10**np.linspace(5,-2,100)*0.5
l1_ratios = np.linspace(.05, 1, 20)
model = linear_model.ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios)
model.fit(train_data, train_y)
prd_y = model.predict(test_data)
print( sm.mean_squared_error(test_y, prd_y))