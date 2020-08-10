#import libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
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
print(tx_cases)
#indicate number of features
n_steps = 3
# split into samples
X, y = prepare_data(tx_cases, n_steps)

#inspect the shape of X
#(number of rows, number of columns)
X.shape

# X is in 2 dimensions. For a LSTM model X needs to be in 3 dimensions
#reshape X by adding the dimension 'n_element'
n_element = 1
X = X.reshape((X.shape[0], X.shape[1], n_element))

##Model Building
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_element)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

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
model.predict(x=X)