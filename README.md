# Using Machine Learning to battle COVID-19

`covid_model.ipynb` creates a long short term memory (LSTM) model and performs a univariate time series analysis with Python, Amazon SageMaker, and other AWS technologies (Tensorflow & Keras).

The general workflow is:

•	`Data Preparation`: converts the list of daily number COVID-19 cases into a 3 dimensional array

•	`Model Building`: builds the LSTM model

•	`Model Training` & `Model Evaluation` : selects the LSTM model that best fits the COVID-19 time series data

•	`Model Prediction`: predicts the number of daily COVID-19 cases for the 10 days after July 26, 2020 

This project analyzes time series data collected by Rearc and made publicly available on [Amazon Data Exchange](https://aws.amazon.com/marketplace/pp/prodview-jmb464qw2yg74?qid=1585594883027&sr=0-1&ref_=srh_res_product_title). Our data set spans from Februrary 12, 2020 to July 26, 2020.

The objective of the this project is to predict the daily number of COVID-19 cases in Texas.
