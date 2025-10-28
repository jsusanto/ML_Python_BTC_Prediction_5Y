# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# STEP 2: Load and clean the CSV
df = pd.read_csv('bitcoin_price.csv')

# Clean 'Price' column
df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)

# Convert 'Date' to datetime and sort
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values('Date')

# STEP 3: Prepare data
prices = df[['Price']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences
prediction_window = 60
x_train, y_train = [], []

for i in range(prediction_window, len(scaled_prices)):
    x_train.append(scaled_prices[i - prediction_window:i, 0])
    y_train.append(scaled_prices[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# STEP 4: Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=30, batch_size=32)

# STEP 5: Forecast next 60 months
forecast = []
last_sequence = scaled_prices[-prediction_window:]

for _ in range(60):  # 60 months = 5 years
    input_seq = last_sequence[-prediction_window:].reshape(1, prediction_window, 1)
    next_price = model.predict(input_seq)[0][0]
    forecast.append(next_price)
    last_sequence = np.append(last_sequence, [[next_price]], axis=0)

# Inverse transform forecast
forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# STEP 6: Plot results
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=60, freq='MS')
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': forecast_prices.flatten()})

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
