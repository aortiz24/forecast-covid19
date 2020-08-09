# Team Python Fever battling COVID-19 with Machine Learning for Social Good

Scripts for creating a long short term memory (LSTM) model and performing a univariate time series analysis with Python, Amazon SageMaker, and other AWS technologies (Tensorflow & Keras).

The general workflow is:

•	`covid_data_prep.py`: convert the list of daily number COVID-19 cases into a 3 dimensional array

•	`covid_model.py`: builds and selects the LSTM model that best fits the COVID-19 time series data

•	`covid_prediction.py`: predicts the number of daily COVID-19 cases for the 10 days after July 26

•	`covid_data_visualization.py`: shows the trend line for the original and predicted daily COVID-19 cases 

This project analyzes time series data collected by Rearc and made publicly available on [Amazon Data Exchange](https://aws.amazon.com/marketplace/pp/prodview-jmb464qw2yg74?qid=1585594883027&sr=0-1&ref_=srh_res_product_title). Our data set spans from Februrary 12, 2020 to July 26, 2020.

The objective of the this project is to predict the daily number of COVID-19 cases in Texas.