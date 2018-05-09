import gzip
import numpy
import os
import codecs
import argparse
import json
import string
import re
import time
import sys

import tensorflow as tf
sess = tf.Session()

import keras
from keras import optimizers
from keras import backend as K
K.set_session(sess)

from keras import regularizers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

from elasticsearch import Elasticsearch, helpers

class FancyConvolutionNetworkClassifier(object):

    model = Sequential()
    rep_max = -100000.0
    rep_size = 0

    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 64, weight_decay = 1e-4):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Dense(n_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()

class SuperSimpleLSTMClassifier(object):
    model = Sequential()

    def __init__(self, embedding_matrix, max_seq_len, n_classes):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model.add(Embedding(nb_words, embed_dim,
                            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='sigmoid'))