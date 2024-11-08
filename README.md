## WaveNet AI

On this project we are using ML/AI Algorithm grouping by the use cases needed on the application 

1. **Anomaly Detection**

     WaveNet AI have functionality to detect anomaly on Microwave Network, because we are working with Network Data which means the data is timestamp data the algorithm we proposed to use will be using Algorithm that worked on Timeseries data, for the timeseries data Recurrent Neural Network was one of the algorithm can be used and effective, but in Recurrent Neural Network there are one flawless there are Gradient Vanishing Problem, so we are using the LSTM (Long-Short Term Memory) who has short term memory that can store the detail and tackle the Gradient Vanishing Problem, beside that we will combining the LSTM classification for Anomaly with Naive Bayes Algorithm to giving more sophisticated result.
     We proposing the method by analyzing the network log data from the data collector on the network link, the data will be gathered by monitoring system, then they are will be analyzed the data using the algorithm prepared before LSTM-NB, based on the user prompt on Chatbot, then the result will be stored to the predict database and then visualize on the dashboard
    
2. **Predict Traffic Density and Latency**
     WaveNet AI have functionality to analyze the network data, then from the data there will be created the prediction of the Traffic Density and Latency on the network link, this is important because the MW Engineer need to determine there are needed to improve or decrease the link to make sure the link not congestion and impacted the network quality.
   
4. **Predict Inventory Demand**
   WaveNet AI has the functionality to predict the inventory demands, this feature create to help the engineer to planning and stocking the material needed, the material is one of the key for project planning to upgrade or downgrade there are must be the stock can be used, so to optimalizing the buying of the material WaveNet proposing to user Deep Learning Based System using LSTM, data to analyze and forecasting the demands of each material, in this folder is the example algorithm usage to analyze ADRO timeseries stock data and giving the forecast of price in the end.

5.  **Suggestion Optimal Stock**
   WaveNet AI beside forecasting the Inventory Demand there are functionality to suggest the optimal material stock to stored on the warehouse.

6. **Predict Optimal Network Link**


