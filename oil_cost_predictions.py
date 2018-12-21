# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:26:57 2018

@author: Vishal
"""
#importing the libraries for dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
training=pd.read_csv('Crude Oil Prices Daily_Train.csv')

#checking for any missing value
sns.heatmap(training.isnull(),yticklabels=False,cmap='viridis',cbar=False)
#deleting all the null value
training=training.dropna()

train_data=training.iloc[:,1:2].values

#features Extraction

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))
scaled_train_data=scaler.fit_transform(train_data)

#Selecting 90 as the time steps

X_train=[]
y_train=[]
for i in range(90,len(scaled_train_data)):
    X_train.append(scaled_train_data[i-90:i,0])
    y_train.append(scaled_train_data[i,0])

#Converting the data to the numpy array as it is expected by our RNN model
X_train=np.array(X_train)
y_train=np.array(y_train)    

#Reshaping 
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    
#builiding the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Intialising the model
model=Sequential()

#First layer
model.add(LSTM(units=40,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
#Second layer
model.add(LSTM(units=40,return_sequences=True))
model.add(Dropout(0.2))
#Third layer
model.add(LSTM(units=40,return_sequences=True))
model.add(Dropout(0.2))
#Fourth Layer
model.add(LSTM(units=40,return_sequences=False))
model.add(Dropout(0.2))
#Output Layer
model.add(Dense(1))

#compiling the model
model.compile(optimizer='adam',loss='mean_squared_error')

#fitting the model on our dataset
model.fit(X_train,y_train,epochs=50,batch_size=32)

#Making the final dataset for making predictions 

testing=pd.read_csv('Crude Oil Prices Daily_Test.csv')
testing=testing.dropna()
test_data=testing.iloc[:,1:2].values

final_dataset=pd.concat((training['Closing Value'],testing['Closing Value']),axis=0)
input_Data=final_dataset[len(final_dataset)-len(test_data)-90:].values
input_Data=input_Data.reshape(-1,1)
input_Data=scaler.transform(input_Data)

#Getting the stock price of previous 60 days
X_test=[]
for i in range(90,111):
    X_test.append(input_Data[i-90:i,0])
X_test=np.array(X_test)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

#Making prediction
predictions=model.predict(X_test)
predictions=scaler.inverse_transform(predictions)

#Visualising the predictions and original data
plt.plot(predictions,color='red',label='Predicted Value')
plt.plot(test_data,color='blue',label='Original Data')
plt.title('Oil Cost Predictions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()




