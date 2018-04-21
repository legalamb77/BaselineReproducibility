import numpy as np
from keras.preprocessing import sequence
from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras import callbacks

from hyperas import optim
from hyperas.distributions import choice, uniform

def softmax_data_wrapper():
    #(x_train, y_tr), (x_test, y_te) = imdb.load_data(num_words = 20000)
    x_train = np.load('simpleAlgorithms/training_x_RT_seq.npy')
    y_tr = np.load('simpleAlgorithms/training_y_RT_seq.npy')
    x_test = np.load('simpleAlgorithms/test_x_RT_seq.npy')
    y_te = np.load('simpleAlgorithms/test_y_RT_seq.npy')
    x_train = sequence.pad_sequences(x_train, maxlen=350)
    x_test = sequence.pad_sequences(x_test, maxlen=350)
    y_train = np.zeros((len(y_tr), 2))
    y_test = np.zeros((len(y_te), 2))
    for ind in range(len(y_tr)):
        y_train[ind][int(y_tr[ind])] = 1
    for ind in range(len(y_te)):
        y_test[ind][int(y_te[ind])] = 1
    return x_train, y_train, x_test, y_test

def model_wrapper(x_train, y_train, x_test, y_test):
    model = Sequential()
    hiddensize = 100
    model.add(Embedding(22000, hiddensize))
    model.add(LSTM(hiddensize))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss = 'binary_crossentropy',
            optimizer = Adam(lr=0.0001),
            metrics = ['accuracy'])
    backs = [callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    model.fit(x_train, y_train, 
            batch_size = 32,
            callbacks = backs,
            epochs = 10,
            validation_split = 0.15,
            shuffle = True)
    score, acc = model.evaluate(x_test, y_test, verbose = 0)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__=='__main__':
    #best_run, best_model = optim.minimize(model = model_wrapper,
    #                        data = softmax_data_wrapper,
    #                        algo = tpe.suggest,
    #                        max_evals = 15,
    #                        trials = Trials())
    X_train, Y_train, X_test, Y_test = softmax_data_wrapper()
    print(X_train)
    print(Y_train)
    model = model_wrapper(X_train, Y_train, X_test, Y_test)
    print(model['model'].evaluate(X_test, Y_test))
