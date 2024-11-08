import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Generate synthetic time series data
np.random.seed(42)
time_steps = 1000
normal_data = np.sin(np.linspace(0, 50, time_steps)) + 0.1 * np.random.randn(time_steps)
# Add anomalies to the dataset
anomalies = np.random.uniform(low=-3, high=3, size=(50,))
anomaly_indices = np.random.randint(0, time_steps, 50)
time_series_data = normal_data.copy()
time_series_data[anomaly_indices] += anomalies

# Step 2: Prepare data for LSTM
def create_dataset(data, window_size=50):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 50
X, y = create_dataset(time_series_data, window_size)

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y = scaler.transform(y.reshape(-1, 1))

# Step 3: Train an LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
model.fit(X_reshaped, y, epochs=10, batch_size=32, validation_split=0.2)

# Step 4: Make predictions and calculate reconstruction error
predictions = model.predict(X_reshaped)
reconstruction_error = np.abs(predictions.flatten() - y.flatten())

# Step 5: Use Naive Bayes to classify anomalies
threshold = np.percentile(reconstruction_error, 95)
labels = (reconstruction_error > threshold).astype(int)

# Step 6: Train Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(reconstruction_error.reshape(-1, 1), labels)

# Step 7: Predict anomalies using Naive Bayes
predictions_nb = nb_model.predict(reconstruction_error.reshape(-1, 1))

# Step 8: Visualization
plt.figure(figsize=(12, 6))
plt.plot(time_series_data, label='Original Data', color='blue')
plt.scatter(np.arange(len(time_series_data))[labels == 1],
            time_series_data[labels == 1], color='red', label='Anomalies')
plt.title('Anomaly Detection using LSTM-NB')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
