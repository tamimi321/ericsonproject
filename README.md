<<<<<<< HEAD
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

=======
## WaveNet AI
<<<<<<< HEAD
>>>>>>> e94c91b13a3e4ea007ab43a228e8968aabf1649f
=======

### Currently on this Repository is the Example of the Algorithm will be used, the developmen will be started base on the example that attached on the repositories, this repositoru will be updated in the future and stored the WaveNet AI project data

On this project we are using ML/AI Algorithm grouping by the use cases needed on the application 

<<<<<<< HEAD
  1. Anomaly Detection
  2. Predict Traffic Density and Latency
  3. Predict Inventory Demand
  4. Suggestion Optimal Stock
>>>>>>> 9b295d533590e24227889b6d798444ac13044789
=======
1. **Anomaly Detection**

     WaveNet AI have functionality to detect anomaly on Microwave Network, because we are working with Network Data which means the data is timestamp data the algorithm we proposed to use will be using Algorithm that worked on Timeseries data, for the timeseries data Recurrent Neural Network was one of the algorithm can be used and effective, but in Recurrent Neural Network there are one flawless there are Gradient Vanishing Problem, so we are using the LSTM (Long-Short Term Memory) who has short term memory that can store the detail and tackle the Gradient Vanishing Problem, beside that we will combining the LSTM classification for Anomaly with Naive Bayes Algorithm to giving more sophisticated result.
     We proposing the method by analyzing the network log data from the data collector on the network link, the data will be gathered by monitoring system, then they are will be analyzed the data using the algorithm prepared before LSTM-NB, based on the user prompt on Chatbot, then the result will be stored to the predict database and then visualize on the dashboard
    
2. **Predict Traffic Density and Latency**

    WaveNet AI have functionality to analyze the network data, then from the data there will be created the prediction of the Traffic Density and Latency on the network link, this is important because the MW Engineer need to determine there are needed to improve or decrease the link to make sure the link not congestion and impacted the network quality.
   
4. **Predict Inventory Demand**

    WaveNet AI has the functionality to predict the inventory demands, this feature create to help the engineer to planning and stocking the material needed, the material is one of the key for project planning to upgrade or downgrade there are must be the stock can be used, so to optimalizing the buying of the material WaveNet proposing to user Deep Learning Based System using LSTM, data to analyze and forecasting the demands of each material, in this folder is the example algorithm usage to analyze ADRO timeseries stock data and giving the forecast of price in the end.

6.  **Suggestion Optimal Stock**

     WaveNet AI beside forecasting the Inventory Demand there are functionality to suggest the optimal material stock to stored on the warehouse.

8. **Predict Optimal Network Link**

   WaveNet AI Functionality to give suggestion, the algorithm usage will be based on the parameter such as altitude, location, distance, etc, we are proposing the machine learning usage to implement the best route can be create to give suggestion to engineer,




>>>>>>> f8e4ae90f3535e20273ec20ab5b54b3011eda37a
