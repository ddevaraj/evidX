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

import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

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
from tqdm import tqdm
from rep_reader import *
from sklearn.metrics import confusion_matrix

import pandas as pd
import random


class SpreadsheetClassificationExecution:

    def __init__(self, sd, embedding_matrix, classifier_type, kerasFile) :

        #training params
        batch_size = 256
        num_epochs = 50

        #model parameters
        num_filters = 64
        embed_dim = 100
        weight_decay = 1e-4

        if classifier_type == 'SuperSimpleLSTMClassifier':
            classifier = SuperSimpleLSTMClassifier(embedding_matrix, sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'FancyConvolutionNetworkClassifier':
            classifier = FancyConvolutionNetworkClassifier(embedding_matrix, sd.max_seq_len, sd.n_classes)
        else:
            raise ValueError("Incorrect Classifier Type: %s"%(classifier_type))

        if kerasFile is not None:
            model_path = kerasFile
            if os.path.exists(model_path):
                classifier = load_model(model_path)
            else:
                hist = classifier.fit(sd.x_train, sd.y_train, batch_size=batch_size,
                            epochs=num_epochs, validation_split=0.1,
                            shuffle=True, verbose=2)
                classifier.save(model_path)
        else:
            hist = classifier.fit(sd.x_train, sd.y_train, batch_size=batch_size,
                             epochs=num_epochs, validation_split=0.1,
                             shuffle=True, verbose=2)

        score = classifier.evaluate(sd.x_test, sd.y_test, verbose=1)
        self.loss = score[0]
        self.accuracy = score[1]

        y_pred = classifier.predict_classes(sd.x_test)
        self.cnf_matrix = confusion_matrix(np.argmax(sd.y_test, axis=1), y_pred)


class SpreadsheetData:

    x_train = None
    x_test = None
    n_classes = 0
    y_train = None
    y_test = None

    def __init__(self, inFile, textColumn, labelColumn, testSize, randomizeTestSet=False):

        df = pd.read_csv(inFile, sep='\t', header=0, index_col=0)
        n_rec = df.shape[0]

        if randomizeTestSet :
            test_ids = sorted(random.sample(range(n_rec), int(testSize)))
        else:
            test_ids = range(int(testSize))
        train_ids = []
        for i in range(n_rec):
            if i not in test_ids:
                train_ids.append(i)

        df_train = df.iloc[train_ids,:]
        df_test = df.iloc[test_ids,:]

        labels = df[labelColumn].unique().tolist()

        y_train_base = [labels.index(i) for i in df_train[labelColumn]]
        y_test_base = [labels.index(i) for i in df_test[labelColumn]]

        # analyze word distribution
        df_train['doc_len'] = df_train[textColumn].apply(lambda words: len(words.split(" ")))
        self.mean_seq_len = np.round(df_train['doc_len'].mean()).astype(int)
        self.max_seq_len = np.round(df_train['doc_len'].mean() + df_train['doc_len'].std()).astype(int)

        np.random.seed(0)

        self.MAX_NB_WORDS = 100000
        nltk_tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

        raw_docs_train = df_train[textColumn].tolist()
        raw_docs_test = df_test[textColumn].tolist()

        print("pre-processing train data...")
        processed_docs_train = []
        all_processed_docs = []
        for doc in tqdm(raw_docs_train):
            tokens = nltk_tokenizer.tokenize(doc)
            filtered = [word for word in tokens if word not in stop_words]
            processed_docs_train.append(" ".join(filtered))
        #end for

        processed_docs_test = []
        for doc in tqdm(raw_docs_test):
            tokens = nltk_tokenizer.tokenize(doc)
            filtered = [word for word in tokens if word not in stop_words]
            processed_docs_test.append(" ".join(filtered))
        #end for

        print("tokenizing input data...")
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=True, char_level=False)
        tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
        word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
        word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
        self.word_index = tokenizer.word_index
        print("dictionary size: ", len(self.word_index))

        #pad sequences
        self.x_train = sequence.pad_sequences(word_seq_train, maxlen=self.max_seq_len)
        self.x_test = sequence.pad_sequences(word_seq_test, maxlen=self.max_seq_len)

        self.n_classes = len(labels)
        self.y_train = keras.utils.to_categorical(y_train_base, num_classes=self.n_classes)
        self.y_test = keras.utils.to_categorical(y_test_base, num_classes=self.n_classes)


class FancyConvolutionNetworkClassifier(Sequential):

    rep_max = -100000.0
    rep_size = 0

    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 64, weight_decay = 1e-4):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self = Sequential()
        self.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.add(MaxPooling1D(2))
        self.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.add(GlobalMaxPooling1D())
        self.add(Dropout(0.5))
        self.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Dense(n_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.summary()

class SuperSimpleLSTMClassifier(Sequential):

    def __init__(self, embedding_matrix, max_seq_len, n_classes):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self = Sequential()
        self.add(Embedding(nb_words, embed_dim,
                            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.add(LSTM(128))
        self.add(Dropout(0.5))
        self.add(Dense(n_classes, activation='sigmoid'))
        self.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        self.summary()

