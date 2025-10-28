# ML_Python_BTC_Prediction_5Y
Machine Learning Bitcoin Prediction in the next 5 years

# What Algorithm Is Being Used?
LSTM (Long Short-Term Memory) Neural Network

LSTM is a type of Recurrent Neural Network (RNN) that’s designed to learn from sequential data — like stock prices, weather patterns, or in your case, Bitcoin prices over time.

# Why Not Regular Neural Networks?
Traditional neural networks (like feedforward or CNNs) treat each input independently. But time-series data has temporal dependencies — what happened yesterday affects today.

# How LSTM Works
LSTM units have memory cells that can remember or forget information over long sequences.

They use gates:
* Forget gate: decides what past info to discard
* Input gate: decides what new info to store
* Output gate: decides what to pass to the next layer

# Why Use LSTM for Bitcoin Price Prediction?

Reason	Explanation
* Time-dependent data:	Bitcoin prices change over time — LSTM learns from past sequences
* Handles long-term dependencies:	Captures patterns across weeks, months, or years
* Learns nonlinear trends:	Bitcoin is volatile — LSTM adapts to complex patterns
* Proven in finance:	Widely used for stock, crypto, and forex forecasting

# How to run
Install Packages

``` pip install -r requirements.txt ```

Run the script

``` python bitcoin_predict.py ```

<img width="1489" height="766" alt="image" src="https://github.com/user-attachments/assets/1517947a-dc70-461b-a976-fed115920a0a" />


# Explanation of the Script
<b> Import Libraries </b>
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
```
* pandas: for loading and manipulating the CSV data.
* numpy: for numerical operations.
* matplotlib: for plotting the results.
* MinMaxScaler: scales prices between 0 and 1 for better neural network performance.
* Sequential, LSTM, Dropout, Dense: components of the LSTM model from TensorFlow/Keras.

<b> Load and Clean the CSV </b>
```
df = pd.read_csv('bitcoin_price.csv')
df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values('Date')
```
* Loads your dataset.
* Cleans the Price column by removing commas and converting to float.
* Converts Date strings to datetime objects.
* Sorts the data chronologically.

<b> Prepare Data for LSTM </b>
```
prices = df[['Price']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)
```
* Extracts the Price column as a NumPy array.
* Scales prices to a 0–1 range using MinMaxScaler (important for neural networks).

<b> Create Training Sequences </b>
```
prediction_window = 60
x_train, y_train = [], []

for i in range(prediction_window, len(scaled_prices)):
    x_train.append(scaled_prices[i - prediction_window:i, 0])
    y_train.append(scaled_prices[i, 0])
```
* Creates input sequences of 60 consecutive prices to predict the next one.
* x_train: 60-day windows.
* y_train: the price following each window.

```
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
```
* Converts lists to NumPy arrays.
* Reshapes x_train to 3D format: [samples, time steps, features] — required by LSTM.

<b> Build the LSTM Model </b>
```
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
```
* LSTM(100): 100 memory units to learn patterns.
* Dropout(0.2): randomly disables 20% of neurons to prevent overfitting.
* Dense(1): final output layer predicting one price.

```
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=30, batch_size=32)
```
* Compiles the model using the Adam optimizer and MSE loss.
* Trains for 30 epochs with batches of 32 samples.

<b> Forecast Next 60 Months </b>
```
forecast = []
last_sequence = scaled_prices[-prediction_window:]
```
* Starts with the last 60 days of scaled prices.

```
for _ in range(60):
    input_seq = last_sequence[-prediction_window:].reshape(1, prediction_window, 1)
    next_price = model.predict(input_seq)[0][0]
    forecast.append(next_price)
    last_sequence = np.append(last_sequence, [[next_price]], axis=0)
```
* Predicts one month at a time.
* Each new prediction is added to the sequence for the next forecast.
* Repeats 60 times to forecast 5 years (monthly).

<b> Convert and Plot Forecast </b>
```
forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
```
* Converts scaled predictions back to actual price values.

```
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=60, freq='MS')
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': forecast_prices.flatten()})
```
* Generates future monthly dates.
* Creates a DataFrame with predicted prices.

```
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Historical Price', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Predicted Price'], label='Forecast (Next 5 Years)', color='red')
plt.title('Bitcoin Price Forecast (Next 5 Years)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
* Plots historical and predicted prices.
* Blue = actual, Red = forecast.

# Summary
LSTM:	Learns patterns in time-series data
MinMaxScaler:	Normalizes prices for better training
Dropout:	Prevents overfitting
Sequential Forecasting:	Predicts one month at a time for 5 years
Plotting:	Visualizes historical and future prices
