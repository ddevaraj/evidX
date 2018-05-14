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
from tqdm import tqdm
import numpy as np

from elasticsearch import Elasticsearch, helpers

class RepReader(object):

    rep_min = 100000.0
    rep_max = -100000.0
    rep_size = 0
    MAX_NB_WORDS = 100000

    def __init__(self, embedding_file=None, index_name=None, elastic=False):
        self.elastic = elastic
        if( elastic ):
            self.es = Elasticsearch()
        
        self.skip_patterns = []
        self.skip_patterns.append( re.compile('^\<.*\>$') )
        
        self.word_rep = {}
            
        if( elastic and embedding_file is not None) :
            self.build_representation_elastic_index(embedding_file, index_name)
        elif(embedding_file is not None): 
            for x in gzip.open(embedding_file):
                x_parts = x.strip().split()
                if len(x_parts) == 2:
                    continue
                word = x_parts[0]
                vec = numpy.asarray([float(f) for f in x_parts[1:]])
                self.word_rep[word] = vec
            #self.word_rep = {x.split()[0]: numpy.asarray([float(f) for f in x.strip().split()[1:]]) for x in gzip.open(embedding_file)}
            self.rep_min = min([x.min() for x in self.word_rep.values()])
            self.rep_max = max([x.max() for x in self.word_rep.values()])
            self.rep_shape = self.word_rep.values()[0].shape
            self.numpy_rng = numpy.random.RandomState(12345)
        else:
            self.elastic = True
            meta = self.es.search(index=index_name,doc_type=['meta'],
                body={"query": {
                    "match_all": {}
                }})
            meta_dict = meta['hits']['hits'][0]['_source']
            self.rep_min = 0.0 #float(meta_dict['rep_min'])
            self.rep_max = 1.0 #float(meta_dict['rep_max'])
            self.rep_shape = int(meta_dict['rep_shape']),
            self.numpy_rng = numpy.random.RandomState(12345)

    def get_word_rep(self, index_name, word):

        w = self.preprocess_word_rep(word)

        if w in self.word_rep:
            rep = self.word_rep[w]

        # Use elastic search index if available.
        elif (self.elastic):
            rep_res = self.es.search(index=index_name, doc_type=['rep'],
                                     body={"query": {
                                         "term": {"word": w}
                                     }})
            try:
                rep = rep_res['hits']['hits'][0]['_source']['rep']
                self.word_rep[w] = rep
            except Exception:
                rep = self.numpy_rng.uniform(low=self.rep_min, high=self.rep_max, size=self.rep_shape)
                self.word_rep[w] = rep

        else:
            rep = self.numpy_rng.uniform(low=self.rep_min, high=self.rep_max, size=self.rep_shape)
            self.word_rep[w] = rep

        return numpy.asarray(rep)


    def get_clause_rep(self, index_name, clause):
        reps = []
        for word in clause.split():
            
            w = self.preprocess_word_rep(word) 
            
            if w in self.word_rep:
                rep = self.word_rep[w]
            
            # Use elastic search index if available. 
            elif( self.elastic ):
                rep_res = self.es.search(index=index_name,doc_type=['rep'],
                    body={"query": {
                        "term" : { "word" : w }
                    }})
                try:
                    rep = rep_res['hits']['hits'][0]['_source']['rep']
                    self.word_rep[w] = rep
                except Exception:
                    rep = self.numpy_rng.uniform(low = self.rep_min, high = self.rep_max, size = self.rep_shape)
                    self.word_rep[w] = rep
                    
            else:
                rep = self.numpy_rng.uniform(low = self.rep_min, high = self.rep_max, size = self.rep_shape)
                self.word_rep[w] = rep
                
            reps.append(rep)
                
        return numpy.asarray(reps)

    def preprocess_word_rep(self, w):
        if( w == 'exLink'):
            w = 'article'
        
        return w
    
#        for p in self.skip_patterns:
#            if re.match(p, w) :
#                return None
#        w = re.sub(ur"\p{P}+", "", w)
#        if len(w) == 0 :
#            return None

    def decode_ref_file(self, embedding_file):
        #from gensim.models import word2vec
        #model = word2vec.Word2Vec.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)
        #model.save_word2vec_format('path/to/GoogleNews-vectors-negative300.txt', binary=False)

        start = time.time()
        for i, x in enumerate(gzip.open(embedding_file)):
            x_parts = x.decode('UTF-8').strip().split()
            if len(x_parts) == 2:
                    continue
    
            w = self.preprocess_word_rep(x_parts[0])
            if w is None:
                continue
            vec = [ float(x) for x in x_parts[1:] ]

            es_fields_keys = ('word', 'rep')
            es_fields_vals = (w, vec)
            
            # Use Global variables to set maxima / minima,
            # TODO: Find a better way
            minimum = min(float(x) for x in x_parts[1:])
            if( minimum < RepReader.rep_min):
                RepReader.rep_min = minimum
            maximum = max(float(x) for x in x_parts[1:])
            if( maximum > RepReader.rep_max):
                RepReader.rep_max = maximum
                            
            # We return a dict holding values from each line
            es_d = dict(zip(es_fields_keys, es_fields_vals))

            if( i%100000 == 0 ):
                print("it: " + str(i) + ", t=" + str(time.time()-start) + " s")

            # Return the row on each iteration
            yield i, es_d     # <- Note the usage of 'yield'

    def build_representation_elastic_index(self, embedding_file, index_name):
        
        self.es.indices.delete(index=index_name, ignore=[400, 404])

        index_exists = self.es.indices.exists(index=[index_name],ignore=404)
        
        if( index_exists is False ):

            rep_min = 10000
            rep_max = -10000  
            shape = 0
    
            i=0
            count=0
            length = 0
            start = time.time()
            
            for x in gzip.open(args.repfile):
                x_parts = x.strip().split()

                if( len(x_parts) == 2 ):
                    count = x_parts[0]
                    shape = x_parts[1]
                    break

            self.es.indices.create(index=index_name, ignore=400)
            
            # Mapping to make the encoding of individual words unique.
            mapping_body = {
                "properties" : {
                    "word" : {
                        "type" : "string",
                        "index" : "not_analyzed" 
                    }
                }
            }
            self.es.indices.put_mapping("rep", mapping_body, index_name)
            
            # NOTE the (...) round brackets. This is for a generator.
            gen = ({
                            "_index": index_name,
                            "_type" : "rep",
                            "_id"     : i,
                            "_source": es_d,
                     } for i, es_d in self.decode_ref_file(embedding_file))
            helpers.bulk(self.es, gen)
            
            actions = [{
                "_index": index_name,
                "_type": "meta",
                "_id": 0,
                "_source": {
                    "rep_shape": str(shape),
                    "rep_count": str(count)
                }
            }]      
            print(actions)
            helpers.bulk(self.es, actions)
        
        meta = self.es.search(index=index_name,doc_type=['meta'],
                body={"query": {
                    "match_all": {}
                }})
        
        # Note that if we've just built the index, it doesn't immediately provide a response
        # So we search and wait until it provides data. 
        while(len(meta['hits']['hits']) == 0) :
            time.sleep(5)
            meta = self.es.search(index=index_name,doc_type=['meta'],
                body={"query": {
                    "match_all": {}
                }}) 
        
        meta_dict = meta['hits']['hits'][0]['_source']
        #self.rep_min = float(meta_dict['rep_min'])
        #self.rep_max = float(meta_dict['rep_max'])
        self.rep_shape = int(meta_dict['rep_shape'].decode('UTF-8')),
        self.numpy_rng = numpy.random.RandomState(12345)

    def read_embedding_matrix_for_vocabulary(self, word_index):

        # embedding matrix
        print('preparing embedding matrix...')
        words_not_found = []
        nb_words = min(self.MAX_NB_WORDS, len(word_index) + 1)
        embed_dim = self.rep_shape[0]
        embedding_matrix = np.zeros((nb_words, embed_dim))
        for word, i in tqdm(word_index.items()):
            if i >= nb_words:
                continue
            embedding_vector = self.get_word_rep(self.es, word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)
        print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        return embedding_matrix


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run LSTM discourse tagger")
    argparser.add_argument('--repfile', metavar='REP-FILE', type=str, help="Gzipped embedding file")
    argparser.add_argument('--indexname', metavar='INDEX-NAME', type=str, help="Name of index for embeddings file")

    args = argparser.parse_args()
    
    repreader = RepReader(args.repfile, args.indexname, elastic=True)

    