# Weather Temperature-Prediction A-Simple RNN Model

A recurrent neural network (Keras SimpleRNN) that predicts **next-day temperature** from the previous 14 days of weather data, and forecasts the following **7 days** recursively.

## Overview
This project walks through a complete time-series regression workflow: data exploration and cleaning, daily aggregation, scaling, sequence windowing, chronological train/validation/test splitting, RNN training, evaluation against a naive baseline, and multi-day forecasting.

**Dataset:** [Weather History, Szeged, Hungary (Kaggle)](https://www.kaggle.com/datasets/muthuj7/weather-dataset),

## Pipeline
1. **Exploration:** first rows, summary statistics, missing-value check, temperature trend and feature plots, correlations.
2. **Cleaning:** physically impossible zero readings (pressure and humidity) treated as missing, hourly data resampled to daily means, incomplete days dropped and gaps filled by time interpolation.
3. **Preprocessing:** `MinMaxScaler` fitted on the training period only (no data leakage), 14-day sliding windows of temperature, humidity and wind speed, target = next day's temperature.
4. **Split:** chronological 70% train / 15% validation / 15% test (no shuffling before the split).
5. **Model:** `Input(14×3) → SimpleRNN(64) → Dropout(0.2) → Dense(1, linear)`, trained with MSE loss, Adam optimizer and an MAE metric (batch size 32, up to 100 epochs, early stopping on validation loss).
6. **Evaluation:** RMSE, MAE and R² on the test set in °C, compared with a "tomorrow = today" persistence baseline. Predicted-vs-actual, scatter and residual plots.
7. **Forecast:** recursive 7-day temperature forecast plotted against recent history. Future humidity and wind are assumed equal to their last-7-day average, so uncertainty grows with each forecast day.

## Results
Fill in after running the notebook:

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| SimpleRNN | 1.878 | 1.415 | 0.951 |
| Persistence baseline | 2.035 | 1.485 | 0.942 |




## Limitations and future work
- SimpleRNN struggles with long-range dependencies, so LSTM and GRU variants are natural next steps.
- Compare sequence lengths (7 vs 14 days).

## Libraries used
Python · TensorFlow/Keras · scikit-learn · pandas · Matplotlib

`Google Colab` : https://colab.research.google.com/drive/1d52ofb5s7Ju0sNxvvUujcFgdC4ckBSoa?usp=sharing
