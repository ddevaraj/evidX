from __future__ import print_function, division
import numpy as np
import pandas as pd

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
import random

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics import confusion_matrix

from rep_reader import RepReader

import matplotlib.pyplot as plt
from tqdm import tqdm
import os, re, csv, math, codecs
import datetime
import argparse
import itertools

def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-' + name[:1],
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('-n' + name[:1], '--no' + name, dest=name, action='store_false')

def get_input_fn(data_set, features, label, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in features}),
        y=pd.Series(data_set[label].values),
        num_epochs=num_epochs,
        shuffle=shuffle)

if __name__ == '__main__':

    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    #parser.add_argument('-b', '--baseDir', help='Base directory for all the files')
    #parser.add_argument('-e', '--esIndex', help='ElasticSearch Index Name')
    '''
    add_boolean_argument(parser, 'full_text_pdf')
    add_boolean_argument(parser, 'refs_flag')
    add_boolean_argument(parser, 'oa')
    add_boolean_argument(parser, 'pmc')
    add_boolean_argument(parser, 'sd')
    add_boolean_argument(parser, 'doi')
    '''
    args = parser.parse_args()

    base_dir = '/Users/Gully/Documents/Projects/2_active/corpora_local/intact/2018-04-17-cleanup/'
    index_name = 'oa_all_fasttext'
    model_file_name = 'i_meth_label.model.h5'
    rep_reader = RepReader(index_name=index_name,elastic=True)
    # From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/input_fn/boston.py

    COLUMNS = ["ID", "i_meth", "p_meth", "pmid", "subfig", "text"]
    FEATURES = ["text"]
    LABEL = "p_meth"

    interaction_df = pd.read_csv(
        base_dir + 'ontologies/i_meth_codes.tsv',
        sep='\t', names=['text','uri','label'],index_col=0)
    interaction_df

    participant_df = pd.read_csv(
        base_dir + 'ontologies/p_meth_codes.tsv',
        sep='\t', names=['text','uri','label'],index_col=0)
    participant_df

    df = pd.read_csv(
        base_dir + 'oa/intact_records_and_captions.tsv',
        sep='\t', names=COLUMNS, skiprows=1, index_col=0)

    df['i_meth_label'] = interaction_df.loc[df['i_meth']]['label'].tolist()
    df['p_meth_label'] = participant_df.loc[df['p_meth']]['label'].tolist()

    df.head(5)

    print("\nInteraction Methods")
    groups_i_meth = df.groupby('i_meth_label')
    for name,group in groups_i_meth :
        print('{:d}: {:s}'.format(len(group),name))

    print("\nParticipant Detection Methods")
    groups_p_meth = df.groupby('p_meth_label')
    for name,group in groups_p_meth :
        print('{:d}: {:s}'.format(len(group),name))

    n_rec = df.shape[0]
    test_set_size = 400

    #test_ids = sorted(random.sample(range(n_rec), test_set_size))
    test_ids = range(test_set_size)
    train_ids = []
    for i in range(n_rec):
        if i not in test_ids:
            train_ids.append(i)

    df_train = df.iloc[train_ids,:]
    df_test = df.iloc[test_ids,:]

    labels_i_meth = df['i_meth_label'].unique().tolist()
    labels_p_meth = df['p_meth_label'].unique().tolist()

    y_train_i_meth = [labels_i_meth.index(i) for i in df_train['i_meth_label']]
    y_train_p_meth = [labels_p_meth.index(p) for p in df_train['p_meth_label']]
    y_test_i_meth = [labels_i_meth.index(i) for i in df_test['i_meth_label']]
    y_test_p_meth = [labels_p_meth.index(p) for p in df_test['p_meth_label']]
    '''
    labels_i_meth = df['i_meth'].unique().tolist()
    labels_p_meth = df['p_meth'].unique().tolist()

    y_train_i_meth = [labels_i_meth.index(i) for i in df_train['i_meth']]
    y_train_p_meth = [labels_p_meth.index(p) for p in df_train['p_meth']]
    y_test_i_meth = [labels_i_meth.index(i) for i in df_test['i_meth']]
    y_test_p_meth = [labels_p_meth.index(p) for p in df_test['p_meth']]
    '''
    #print(labels_i_meth_s.loc[df_train['i_meth_label']])

    #print("\nParticipant Detection Methods")
    #groups_p_meth = df.groupby('p_meth_label')
    #labels_p_meth = []
    #for name,group in groups_p_meth :
    #    print('{:d}: {:s}'.format(len(group),name))
    #    labels_p_meth.append(name)

    #n_classes

    #load embeddings
    #print('loading word embeddings...')
    #embeddings_index = {}
    #f = codecs.open('/Users/Gully/Documents/Projects/2_active/corpora_local/embeddings_pubmed_files/all_text.txt.model.vec', encoding='utf-8')
    #for line in tqdm(f):
    #    values = line.rstrip().rsplit(' ')
    #    word = values[0].lower()
    #    coefs = np.asarray(values[1:], dtype='float32')
    #    embeddings_index[word] = coefs
    #f.close()
    #print('found %s word vectors' % len(embeddings_index))

    #visualize word distribution
    df_train['doc_len'] = df_train['text'].apply(lambda words: len(words.split(" ")))
    mean_seq_len = np.round(df_train['doc_len'].mean()).astype(int)
    max_seq_len = np.round(df_train['doc_len'].mean() + df_train['doc_len'].std()).astype(int)
    #sns.distplot(df_train['doc_len'], hist=True, kde=True, color='b', label='doc len')
    #plt.axvline(x=mean_seq_len, color='black', linestyle='--', label='mean')
    #plt.axvline(x=max_seq_len, color='red', linestyle='--', label='mean + std')
    #plt.title('comment length'); plt.legend()
    #plt.show()

    #sns.set_style("whitegrid")
    np.random.seed(0)

    DATA_PATH = '../input/'
    EMBEDDING_DIR = '../input/'

    MAX_NB_WORDS = 100000
    nltk_tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    raw_docs_train = df_train['text'].tolist()
    raw_docs_test = df_test['text'].tolist()

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
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
    word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
    word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    #pad sequences
    x_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    x_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

    n_classes = len(labels_i_meth)
    y_train = keras.utils.to_categorical(y_train_i_meth, num_classes=n_classes)
    y_test = keras.utils.to_categorical(y_test_i_meth, num_classes=n_classes)

    #training params
    batch_size = 256
    num_epochs = 50

    #model parameters
    num_filters = 64
    embed_dim = 100
    weight_decay = 1e-4

    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in tqdm(word_index.items()):
        if i >= nb_words:
            continue
        embedding_vector = rep_reader.get_word_rep(index_name,word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    #print("words in document not found in the index : ", np.random.choice(words_not_found, 10))

    # Model 1 CNN
    '''
    model = Sequential()
    model.add(Embedding(nb_words, embed_dim,
              weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(n_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    '''
    '''
    model = Sequential()
    model.add(Embedding(nb_words, embed_dim,
              weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    #define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping]

    hist = model.fit(x_train, y_train, batch_size=batch_size,
                     epochs=num_epochs, callbacks=callbacks_list,
                     validation_split=0.1, shuffle=True)
    '''
    model = Sequential()
    model.add(Embedding(nb_words, embed_dim,
                        weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='sigmoid'))

    model_path = base_dir + '/' + model_file_name
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=num_epochs, validation_split=0.1,
                    shuffle=True, verbose=2)
        model.save(model_path)

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print(n_classes)
    #word_seq_test.shape

    y_pred = model.predict_classes(x_test)
    y_pred_raw = model.predict(x_test)
    print("gold\tpred\n")
    total = 0
    for g,p in zip(y_test_i_meth,y_pred):
        s = 0
        if g==p :
            s = 1
        print('%d\t%d\t%d'%(g,p,s))
        total = total + s

    print('Accuracy:%f'%(total/len(y_test_i_meth)))
    cnf_matrix = confusion_matrix(y_test_i_meth, y_pred)
    print(cnf_matrix)
