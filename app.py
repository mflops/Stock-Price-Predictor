import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import time

model = keras.models.load_model('Stock_Price_Predictor.keras')

st.header('Stock Price Predictor')

ticker = st.text_input('Enter TICKER', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Add error handling for data download
try:
    data = pd.DataFrame(yf.download(ticker, start, end))
    if data.empty:
        st.error(f"No data found for ticker {ticker}. Please try a different ticker.")
        st.stop()
except Exception as e:
    st.error(f"Error downloading data: {str(e)}")
    st.error("Please try again in a few minutes or use a different ticker.")
    st.stop()

st.subheader('STOCK Data')
st.write(data)

train_data = data.Close[0 : int(len(data)*0.8)]
test_data = data.Close[int(len(data)*0.8) : len(data)]

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = train_data.tail(100)
test_data = pd.concat([past_100_days, test_data], ignore_index=True)

# Validate data before scaling
if len(test_data) == 0:
    st.error("No data available for prediction. Please try a different ticker.")
    st.stop()

test_data_scale = scaler.fit_transform(test_data.values.reshape(-1, 1))

st.subheader('Price vs MA50')
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, 'r')
plt.plot(ma_100, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100, 'r')
plt.plot(ma_200, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x, y = [], []

for i in range(100, test_data_scale.shape[0]):
    x.append(test_data_scale[i-100 : i])
    y.append(test_data_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_

predict = predict*scale
y = y*scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)


