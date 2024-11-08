#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


#Step 2: Load and Preprocess Data
# Load data
df = pd.read_csv('traffic_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Display the first few rows
print(df.head())

# Fill missing values if any
df.fillna(method='ffill', inplace=True)

# Plot traffic density and latency over time
plt.figure(figsize=(12, 6))
plt.plot(df['traffic_density'], label='Traffic Density')
plt.plot(df['latency'], label='Latency')
plt.legend()
plt.title("Traffic Density and Latency")
plt.show()

#Step 3: Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)


#Step 4: Prepare Data for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])
        y.append(data[i + sequence_length, :])
    return np.array(X), np.array(y)

sequence_length = 60  # Number of time steps (e.g., last 60 minutes)
X, y = create_sequences(scaled_data, sequence_length)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")


#Step 5: Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=2))  # Output layer for predicting both traffic density and latency

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


#Step 6: Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Plot training & validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()


#Step 7: Make Predictions
# Predict using the test set
y_pred = model.predict(X_test)

# Reverse scaling to get actual values
y_test_rescaled = scaler.inverse_transform(np.hstack((y_test, np.zeros((y_test.shape[0], 1)))))
y_pred_rescaled = scaler.inverse_transform(np.hstack((y_pred, np.zeros((y_pred.shape[0], 1)))))

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:, 0], label='Actual Traffic Density')
plt.plot(y_pred_rescaled[:, 0], label='Predicted Traffic Density')
plt.plot(y_test_rescaled[:, 1], label='Actual Latency')
plt.plot(y_pred_rescaled[:, 1], label='Predicted Latency')
plt.legend()
plt.title("Traffic Density and Latency Predictions")
plt.show()


#Step 8: Evaluate the Model
from sklearn.metrics import mean_squared_error

# Calculate RMSE for traffic density and latency
rmse_density = np.sqrt(mean_squared_error(y_test_rescaled[:, 0], y_pred_rescaled[:, 0]))
rmse_latency = np.sqrt(mean_squared_error(y_test_rescaled[:, 1], y_pred_rescaled[:, 1]))

print(f"RMSE (Traffic Density): {rmse_density}")
print(f"RMSE (Latency): {rmse_latency}")