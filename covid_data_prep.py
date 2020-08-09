#import packages
import pandas as pd
import numpy as np

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
tx_cases = pd.read_csv("data/Texas_covid_data.csv", usecols=["cases"])
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