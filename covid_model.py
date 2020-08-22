#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import TimeSeriesSplit

##Data Preparation
# preparing independent and dependent features
def prepare_data(time_series_data, n_features):
	X, y =[],[]
	for i in range(len(tx_cases)):
		# find the end of this pattern
		end_ix = i + n_features
		# check if we are beyond the sequence
		if end_ix > len(tx_cases)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = tx_cases[i:end_ix], tx_cases[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

#define input sequence
tx_cases = pd.read_csv("Texas_covid_data.csv", usecols=["cases"])
#perform data normalization 
#daily number of Texas covid cases converted to fit between 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1))
tx_cases_scaled = scaler.fit_transform(tx_cases)
#indicate number of features
n_steps = 3
# split into samples
X, y = prepare_data(tx_cases_scaled, n_steps)

#inspect the shape of X
#(number of rows, number of columns)
print(X.shape)

# X is in 2 dimensions. For a LSTM model X needs to be in 3 dimensions
#reshape X by adding the dimension 'n_element'
n_element = 1
X = X.reshape((X.shape[0], X.shape[1], n_element))

##Model Building
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_element)))
model.add(Dropout(0.2)) #prevents overfitting
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(Dropout(0.2)) #prevents overfitting
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(Dropout(0.2)) #prevents overfitting
model.add(LSTM(50))
model.add(Dropout(0.2)) #prevents overfitting
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

#split X, y variables into training & test data sets for each variable
tscv = TimeSeriesSplit()
print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=5)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

##Model Training
#train using training data set
model.fit(X_train, epochs=30, verbose=1)

##Model Evaluation
#evaluate using test data set
model.evaluate(x=X_test, verbose=1)

##Model Prediction
#predict using numpy array of daily COVID-19 cases in Texas
prediction = model.predict(x=X)
#reverse normalization of prediction data
predictions = scaler.inverse_transform(prediction)

##Data Visualization
#plot actual and predicted data
plt.figure(figsize=(10,6))
plt.plot(tx_cases, color='blue', label='Actual Daily COVID-19 Cases')
plt.plot(predictions, color='red', label='Predicted Daily COVID-19 Cases')
plt.title('Daily COVID-19 Cases Prediction')
plt.xlabel('Date')
plt.ylabel('Daily COVID-19 Cases')
plt.legend()
plt.show()