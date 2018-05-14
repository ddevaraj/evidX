from __future__ import print_function, division

import os, re, csv, math, codecs
import datetime
import argparse
from rep_reader import RepReader
from keras_spreadsheet_classifier import *
from tqdm import tqdm
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

if __name__ == '__main__':

    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()

    parser.add_argument('inFile', help='Input File')
    parser.add_argument('textColumn', help='Name of text column')
    parser.add_argument('labelColumn', help='Name of text column')
    parser.add_argument('testSize', help='Size of held-out test set')
    parser.add_argument('--kerasFile', help='Keras model file')
    parser.add_argument('--esIndex', help='ElasticSearch Representation Index Name')
    parser.add_argument('--repFile', help='Representation File Path')

    add_boolean_argument(parser, 'randomizeTestSet')

    args = parser.parse_args()

    rep_reader = None
    if args.repFile is not None:
        rep_reader = RepReader(embedding_file=args.repFile, elastic=False)
    elif args.esIndex is not None:
        rep_reader = RepReader(index_name=args.esIndex, elastic=True)
    else:
        raise ValueError("You must specify either kerasFile or esIndex. Neither specified.")

    sd = SpreadsheetData(args.inFile, args.textColumn, args.labelColumn, args.testSize, args.randomizeTestSet)

    # embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(sd.MAX_NB_WORDS, len(sd.word_index) + 1)
    embed_dim = rep_reader.rep_shape[0]
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in tqdm(sd.word_index.items()):
        if i >= nb_words:
            continue
        embedding_vector = rep_reader.get_word_rep(args.esIndex, word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    run = SpreadsheetClassificationExecution(sd, embedding_matrix, "SuperSimpleLSTMClassifier", args.kerasFile)

    print("Accuracy:%f"%run.accuracy)
