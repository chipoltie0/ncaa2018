# -*- coding: utf-8 -*-
"""

KERAS NCAA MODEL VERSION 2
Created on Wed Mar 28 22:00:36 2018

@author: chipo
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import regularizers

data = pd.read_csv('final_bracket_train.csv')

data.drop([])
