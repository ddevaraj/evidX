# EvidX Source Code

Here we put code for the EvidX project pertaining to various tasks.

- text classifiers for figure captions from molecular interaction papers. 

# Two ways of loading word embeddings

This system uses word embeddings which can be loaded directly into memory or from an elastic search index. 
Both functions are accomplished with the `rep_reader.py` module. 

## Building the ES index

Build the ES index by executing the following command:

```
    python rep_reader.py --repfile <path/to/word/embedding/vec/file> --indexname 'elasticsearch-index-name-to-use'
```
This provides a way of running the classfier on smaller machines (e.g., laptops) without having to load the whole embedding file which is helpful for debugging.  

## A prebuilt embedding file

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

# Running the classifier.

## From the embedding file. 
```
python classify_spreadsheet.py --kerasFile /path/to/keras.model.file.h5 --repFile /path/to/embedding_file.vec.gz /path/to/spreadsheet.tsv <text-column-name> <label-column-name> <test_set_size#>
```
so running the system on local data would look like this:
```
python /nas/home/burns/tools/python/evidX/classify_spreadsheet.py --kerasFile /nas/evidx/corpora/intact/2018-04-17-cleanup/oa/p_meth_lstm.model.h5 --repFile /nas/evidx/corpora/molecular_oa_pmc/all_text.txt.model.vec.gz /nas/evidx/corpora/intact/2018-04-17-cleanup/oa/intact_records_and_captions_labels.tsv text p_meth 400
```

## From the elasticseach index. 
```
python classify_spreadsheet.py --kerasFile /path/to/keras.model.file.h5 --esIndex name.of.index /path/to/spreadsheet.tsv <text-column-name> <label-column-name> <test_set_size#>
```

## Output.

The system trains a model, saves it to the specified \*.h5 file, holds out a test set and runs an evaluation on that test set. It prints it to standard output. 
