import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


#''' Date Format --> YYYY-MM-DD '''
START_DATE = '2008-01-01' 
END_DATE = str(dt.datetime.now().strftime('%Y-%m-%d'))


st.title('Stock Trend Prediction')


ticker = st.text_input('Enter Stock Ticker', 'SBIN.NS')
df = yf.Ticker(ticker).history(start = START_DATE, end = END_DATE)


# Describing Data
st.subheader('Data from 2008 - ' + str(dt.datetime.now().strftime('%Y')))
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
# st.legend()
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r', label='100 MOVING AVERAGE')
plt.plot(ma200, 'g', label = '200 MOVING AVERAGE')
plt.plot(df.Close, 'b', label = 'CLOSING PRICE')
plt.legend()
st.pyplot(fig)

#Spliting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

#load LSTM model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
	x_test.append(input_data[i-100: i])
	y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)


#Making Predictions
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Visualizations
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

from streamlit_plotly_events import plotly_events

# fig = px.line(x = [1], y = [1])
# selected_points = plotly_events(fig)

# with st.beta_expander('Plot'):
# 	fig = px.line(x=[1], y=[1])
# 	selected_points = plotly_events(fig)

# fig = fig2.line(x = [1], y = [1])
# selected_points = plotly_events(fig, click_event = False, hover_event = True)