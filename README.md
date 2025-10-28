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

