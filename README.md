## WaveNet AI

On this project we are using ML/AI Algorithm grouping by the use cases needed on the application 

1. **Anomaly Detection**

     WaveNet AI have functionality to detect anomaly on Microwave Network, because we are working with Network Data which means the data is timestamp data the algorithm we proposed to use will be using Algorithm that worked on Timeseries data, for the timeseries data Recurrent Neural Network was one of the algorithm can be used and effective, but in Recurrent Neural Network there are one flawless there are Gradient Vanishing Problem, so we are using the LSTM (Long-Short Term Memory) who has short term memory that can store the detail and tackle the Gradient Vanishing Problem, beside that we will combining the LSTM classification for Anomaly with Naive Bayes Algorithm to giving more sophisticated result.
     We proposing the method by analyzing the network log data from the data collector on the network link, the data will be gathered by monitoring system, then they are will be analyzed the data using the algorithm prepared before LSTM-NB, based on the user prompt on Chatbot, then the result will be stored to the predict database and then visualize on the dashboard
    
3. **Predict Traffic Density and Latency**
4. **Predict Inventory Demand**
5.  **Suggestion Optimal Stock**


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

# Suggestion Optimal Stock

Below is a sample script for calculating optimal stock levels using the Economic Order Quantity (EOQ) model. This model helps determine the most cost-effective amount of stock to order by balancing the ordering costs and holding costs.

Concept: Economic Order Quantity (EOQ)
The formula for EOQ is:

𝐸𝑂𝑄 =(Square root)√2DS/H 
​
Where:
D = Annual demand (units)
S = Ordering cost per order
H = Holding cost per unit per year
The EOQ model assumes constant demand and lead time, which may not fit all situations but is a great starting point for optimizing stock levels. 

# Sample output
Optimal Order Quantity (EOQ) for each product:
     Product  Annual_Demand  Ordering_Cost  Holding_Cost_Per_Unit        EOQ
0  Product A           5000             50                    2.5  200.000000
1  Product B           3000             75                    3.0  223.606798
2  Product C           8000             60                    2.0  346.410162

## Explanation
Input Data: The script uses sample data for three products, including annual demand, ordering costs, and holding costs.
EOQ Calculation: The script calculates the EOQ for each product using the formula provided.
Output: Displays the EOQ for each product, which indicates the optimal number of units to order to minimize total inventory costs.
## Advanced Extensions
Demand Forecasting: If demand is not constant, use time series forecasting models (e.g., ARIMA, Prophet) to predict future demand.
Stockout Costs: Incorporate stockout costs if stockouts are costly for your business.
Lead Time: Adjust calculations if there are significant lead times to consider.
