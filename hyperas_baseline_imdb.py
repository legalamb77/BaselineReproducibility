from keras.preprocessing import sequence
from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras import callbacks

from hyperas import optim
from hyperas.distributions import choice, uniform#, conditional

max_features = 20000
maxlen = 350
batch_size = 64

def data_wrapper():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 20000)
    x_train = sequence.pad_sequences(x_train, maxlen=350)
    x_test = sequence.pad_sequences(x_test, maxlen=350)
    return x_train, y_train, x_test, y_test

def model_wrapper(x_train, y_train, x_test, y_test):
    model = Sequential()
    hiddensize = {{choice([64, 128, 200, 300])}}
    model.add(Embedding(20000, hiddensize))
    model.add(LSTM(hiddensize, dropout={{uniform(0,1)}}, recurrent_dropout={{uniform(0,1)}}))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy',
            optimizer = {{choice(['rmsprop', Adam(lr=0.0001)])}},
            metrics = ['accuracy'])
    backs = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    model.fit(x_train, y_train, 
            batch_size = {{choice([64, 128])}},
            callbacks = backs,
            epochs = 10,
            validation_split = 0.15)
    score, acc = model.evaluate(x_test, y_test, verbose = 0)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__=='__main__':
    best_run, best_model = optim.minimize(model = model_wrapper,
                            data = data_wrapper,
                            algo = tpe.suggest,
                            max_evals = 15,
                            trials = Trials())
    X_train, Y_train, X_test, Y_test = data_wrapper()
    print("Best eval: ")
    print(best_model.evaluate(X_test, Y_test))
    print("Best parameters: ")
    print(best_run)
