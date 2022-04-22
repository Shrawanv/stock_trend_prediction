# import libraries
import math
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt


START_DATE = '2008-01-01' 
END_DATE = str(dt.datetime.now().strftime('%Y-%m-%d'))

st.title('Stock Price Prediction')

# Get the stock quote
ticker = st.text_input('Enter Stock Ticker', 'SBIN.NS')
df = df = yf.Ticker(ticker).history(start = START_DATE, end = END_DATE)


# Describing data
st.subheader('Data from 2008 - '+ str(dt.datetime.now().strftime('%Y')))
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label = 'Moving Average 100')
plt.plot(df.Close, label = 'Closing Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label='100 MOVING AVERAGE')
plt.plot(ma200, label = '200 MOVING AVERAGE')
plt.plot(df.Close, label = 'CLOSING PRICE')
plt.legend()
st.pyplot(fig)


# Create new data frame with only Close column
data = df.filter(['Close'])
#convert the dataframe into numpy array
dataset = data.values
#Get the number of rows to train the model
training_data_len = math.ceil(len(dataset)*  .8)

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

# Create testing data set
#creat a new array containing scaled data set 
test_data = scaled_data[training_data_len - 60: :]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len: :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#convert the array to numpy array 
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#load LSTM model
model = load_model('keras_modelv2.h5')

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
st.subheader('Prediction Price with compare to Original Price')
fig2 = plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price in INR', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'])
st.pyplot(fig2)

#Get the quote
#Get the quote
start_date = dt.datetime(2008,1,1)
end_date = dt.datetime(2022,4,20)
df = yf.Ticker(ticker).history(start = start_date, end = end_date)
#create new dataframe
new_df = df.filter(['Close'])
#Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an emty list
X_test = []
X_test.append(last_60_days_scaled)
#convert to numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)

st.subheader(f'Predicted Price for {ticker}:')
st.write(f'Predicted Price for {ticker.split(".")[0]} is {pred_price[0][0]:.2f}')
