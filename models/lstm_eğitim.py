import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Klasörleri oluştur
os.makedirs("models/lstm_models", exist_ok=True)

# Veriyi oku
df = pd.read_csv("demand_forecasting_data_cleaned.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Product_ID'] = df['Product_ID'].astype('category')

# Parametreler
look_back = 30
epochs = 50
batch_size = 16

product_ids = df['Product_ID'].unique()

for pid in product_ids:
    data = df[df['Product_ID'] == pid].copy()
    data = data.groupby('Date')['Demand'].sum().reset_index()
    data['Demand'] = data['Demand'].rolling(window=7).mean()
    data.dropna(inplace=True)

    if len(data) <= look_back:
        print(f"Yetersiz veri: {pid}")
        continue

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data['Demand'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    model.save(f"models/lstm_models/{pid}.h5")
    print(f"Model kaydedildi: {pid}.h5")



