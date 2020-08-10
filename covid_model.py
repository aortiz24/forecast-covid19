#import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import TimeSeriesSplit

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

#Model prediction
#predict using numpy array of daily COVID-19 cases in Texas
model.predict(x=X)