## WaveNet AI

On this project we are using ML/AI Algorithm grouping by the use cases needed on the application 

1. **Anomaly Detection**

     WaveNet AI have functionality to detect anomaly on Microwave Network, because we are working with Network Data which means the data is timestamp data the algorithm we proposed to use will be using Algorithm that worked on Timeseries data, for the timeseries data Recurrent Neural Network was one of the algorithm can be used and effective, but in Recurrent Neural Network there are one flawless there are Gradient Vanishing Problem, so we are using the LSTM (Long-Short Term Memory) who has short term memory that can store the detail and tackle the Gradient Vanishing Problem, beside that we will combining the LSTM classification for Anomaly with Naive Bayes Algorithm to giving more sophisticated result.

We proposing the method by analyzing the network log data from the data collector on the network link, the data will be gathered by monitoring system, then they are will be analyzed the data using the algorithm prepared before LSTM-NB, based on the user prompt on Chatbot, then the result will be stored to the predict database and then visualize on the dashboard
    
3. **Predict Traffic Density and Latency**
4. **Predict Inventory Demand**
5.  **Suggestion Optimal Stock**
