#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM, Activation
import matplotlib.pyplot as plt


def train_dense(x, y):
    model = Sequential()
    model.add(Dense(100, input_dim=x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=10, batch_size=1000, shuffle=True, validation_split=0.2, verbose=1)
    model.save('dense_model.h5')


def samples_as_sequences(x, time_window):
    sequences = []
    for i in range(len(x) - time_window):
        sequences.append(x[i:i + time_window])
    return np.array(sequences)


def train_rnn(x, y):
    TIME_WINDOW = 100

    x_train = samples_as_sequences(x, TIME_WINDOW)
    y_train = y[TIME_WINDOW:]

    rnn = Sequential()
    # rnn.add(LSTM(32, return_sequences=True, input_dim=len(train_cols), input_length=TIME_WINDOW))
    # rnn.add(LSTM(32, return_sequences=True))
    # rnn.add(LSTM(32))

    rnn.add(LSTM(32, input_shape=(TIME_WINDOW, x.shape[1]), dropout=0.2, recurrent_dropout=0.2))

    # 1 Output (regression without actiavtion)
    rnn.add(Dense(1))

    rnn.compile(optimizer='adam', loss='mse')
    rnn.fit(x_train, y_train, batch_size=64, epochs=5)
    rnn.save('rnn_model.h5')

    plt.figure(figsize=(16, 9))
    plt.plot(rnn.predict(x_train), 'k-', label='Predicted')
    plt.plot(y_train, 'k--', label='True workload')
    plt.legend()
