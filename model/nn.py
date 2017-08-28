#!/usr/bin/env python

import numpy as np
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM, Activation
from sklearn.preprocessing import MinMaxScaler


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
    x_train = samples_as_sequences(x, time_window)
    y_train = y[time_window:]

    rnn = Sequential()
    rnn.add(LSTM(128, input_shape=(time_window, x.shape[1]), dropout=0.1, recurrent_dropout=0.1))

    # 1 Output (regression without activation)
    rnn.add(Dense(1))

    rnn.compile(optimizer='adam', loss='mse')
    rnn.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=10)

    if save_to is not None:
        rnn.save(save_to)
    return rnn


def fit_to_size(a, l):
    w = np.zeros(l)
    window_size = len(a) / (l + 1)
    for i in range(l):
        w[i] = a[window_size * i: window_size * (i + 1)].mean()
    return w


def train(workload, features, model_path=None, time_window=TIME_WINDOW, predict=True):
    scaler = MinMaxScaler(feature_range=(0, 1))

    x = scaler.fit_transform(features.values)
    y = fit_to_size(workload, len(features))

    model = None
    if model_path is not None and os.path.exists(model_path):
        print('Found already trained model %s, loading' % model_path)
        model = load_model(model_path)
    else:
        print('Training RNN LTSM model')
        model = train_rnn(x, y, time_window=time_window, save_to=model_path)

    if predict:
        predictions_raw = model.predict(samples_as_sequences(x, time_window=time_window)).flatten()
        predictions = np.zeros(len(y))
        predictions[time_window:len(features)] = predictions_raw
        return model, predictions

    return model
