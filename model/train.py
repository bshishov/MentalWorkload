#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM, Activation
import matplotlib.pyplot as plt
import os


TIME_WINDOW = 100
BATCH_SIZE = 100


def train_dense(x, y, save_to=None):
    model = Sequential()
    model.add(Dense(100, input_dim=x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=10, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.2, verbose=1)

    if save_to is not None:
        model.save(save_to)
    return model


def samples_as_sequences(x, time_window):
    sequences = []
    for i in range(len(x) - time_window):
        sequences.append(x[i:i + time_window])
    return np.array(sequences)


def train_rnn(x, y, time_window=TIME_WINDOW, save_to=None):
    x_train = samples_as_sequences(x, TIME_WINDOW)
    y_train = y[TIME_WINDOW:]

    rnn = Sequential()
    rnn.add(LSTM(32, input_shape=(time_window, x.shape[1]), dropout=0.2, recurrent_dropout=0.2))

    # 1 Output (regression without actiavtion)
    rnn.add(Dense(1))

    rnn.compile(optimizer='adam', loss='mse')
    rnn.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=5)

    if save_to is not None:
        rnn.save(save_to)
    return rnn
