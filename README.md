# Anomaly-Detection : LSTM-NB
Ensure you have the following packages installed:
pip install numpy pandas matplotlib scikit-learn tensorflow

# Approach
## Data Preparation:
Generate synthetic time series data containing normal data and anomalies.
LSTM Model:
Train an LSTM-based model to predict the next values in the time series.
Calculate reconstruction errors to identify anomalies.
Naive Bayes Classifier:
Use Naive Bayes to classify the sequences based on their reconstruction errors.
Visualization:
Visualize normal and anomalous data points.

## Running the Script
python anomaly_detection_lstm_nb.py

Expected Output
The script will plot the original time series with anomalies marked in red. The combination of LSTM and Naive Bayes helps in accurately detecting anomalies based on patterns in the time series data.


# Predict Traffic Density and Latency with LSTM
## Prerequisites:
Make sure you have these Python libraries installed:
pip install numpy pandas matplotlib scikit-learn tensorflow

## Explanation
Data Preprocessing: Load the CSV data, handle missing values, and scale it using MinMaxScaler.
Data Preparation: Create sequences of data to feed into the LSTM model.
Model Building: Build an LSTM model with Dense layers for multi-output predictions.
Training: Train the model and visualize training progress.
Prediction and Evaluation: Make predictions and evaluate the model using RMSE.
Notes
Adjust the sequence_length, number of LSTM layers, units, and epochs to optimize model performance.
The dataset should have sufficient data points for meaningful sequence learning by the LSTM model.
You can save and load your trained model using model.save() and load_model() from Keras.

## Running the Script
python app.py

