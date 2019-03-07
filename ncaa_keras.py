# -*- coding: utf-8 -*-
"""
KERAS MODEL

Created on Sun Mar 11 23:34:57 2018

@author: chipo
"""
# no. of inputs: 122
# no. of outputs: 2

import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping
from keras import regularizers

def parse_data(input_file):
    dataframe = pd.read_csv(input_file)
    data = dataframe.values
    X = data[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]]
    Y = data[:,[123]]
    return train_test_split(X,Y)
    
def get_optimizer():
    return SGD(lr=0.01, decay=0.001)

def get_model():
    model = Sequential()

    # Input layer
    model.add(Dense(25,input_shape=(122,),kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',kernel_regularizer=regularizers.l2()))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(25, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',kernel_regularizer=regularizers.l2()))
    model.add(Dropout(0.1))
    
    model.add(Dense(25, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',kernel_regularizer=regularizers.l2()))
    model.add(Dropout(0.1))
    
    model.add(Dense(25, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',kernel_regularizer=regularizers.l2()))
    model.add(Dropout(0.1))
    
    model.add(Dense(25, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',kernel_regularizer=regularizers.l2()))
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(2,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform'))
    optim = get_optimizer()
    model.compile(loss='sparse_categorical_cross_entropy', optimizer=optim, metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, epochs=100, verbose=True, validation_split = 0.3, shuffle=True,callbacks=[])

def test_model(model, X_test, Y_test):
    score = model.evaluate(X_test, Y_test, verbose=True)
    return score
    
def main():
    # YOUR INPUT FILE GOES HERE
    input_file = 'data.csv'
    X_train, X_test, Y_train, Y_test = parse_data(input_file)
    model = get_model()
    train_model(model, X_train, Y_train)
    score = test_model(model, X_test, Y_test)
    model.save('model.hdf5')
    Y_pred = model.predict(X_test)
    for i in range(len(X_test)):
        print(Y_pred[i], Y_test[i])
    print('Score: ',score)
    return score

if __name__ == '__main__':
    main()

