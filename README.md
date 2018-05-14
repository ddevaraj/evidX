# EvidX Source Code 

Here we put preliminary code for the EvidX project pertaining to various tasks.
At present, this is only concerned with developing text classifiers for figure captions from molecular interaction papers, 
but we will provide more structure as we progress. 

# Use of ElasticSearch to index word embeddings

This system uses word embeddings, but requires them to be loaded into an elastic search index (rather than loading them into memory). 
This can be accomplished with the `rep_reader.py` script by executing the following command:

```
    python rep_reader.py --repfile <path/to/word/embedding/vec/file> --indexname 'elasticsearch-index-name-to-use'
```
## Prerequisite data

Here is a gzipped archive of a good molecular biology word embedding file.
It is located on the ISI `NAS` filesystem at the following location:

/nas/evidx/corpora/molecular_oa_pmc/fasttext_models/all_text.txt.model.vec.tar.gz

This was  built by applying `Fasttext` to 
611,539 full text articles and 3,079,497 Medline records that were gathered in response to the following PubMed query:

```
("cells"[MeSH Terms] OR "Multiprotein Complexes"[mh] OR "Protein Aggregates"[mh] OR 
"Hormones, Hormone Substitutes, and Hormone Antagonists"[mh] OR "Enzymes and Coenzymes"[mh] OR 
"Carbohydrates"[mh] OR "Lipids"[mh] OR "Amino Acids, Peptides, and Proteins"[mh] OR 
"Nucleic Acids, Nucleotides, and Nucleosides"[mh] OR "Biological Factors"[mh] OR 
"Pharmaceutical Preparations"[mh] OR "Metabolism"[mh] OR "Cell Physiological Phenomena"[mh] OR
"Genetic Phenomena"[mh])
```
This query was design to use high-level molecular MeSH terms to select only molecular papers from PubMed to keep 
the word embeddings relevant to our domain of study: molecular interactions. 

## Running the classifier.

```
python classify_spreadsheet.py <path/to/spreadsheet> <text-column-name> <label-column-name> <es-index-name> <path/to/saved/keras/model> <test_set_size#>
```
so running the system on local data would look like this:
```
python classify_spreadsheet.py /nas/evidx/corpora/intact/2018-04-17-cleanup/oa/intact_records_and_captions_labels.tsv text i_meth_label \
    oa_all_fasttext /nas/evidx/corpora/intact/2018-04-17-cleanup/oa/i_meth_label_cnn.model.h5 400
```
