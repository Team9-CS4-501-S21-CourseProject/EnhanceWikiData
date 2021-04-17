# language-models-are-knowledge-graphs-pytorch
Language models are open knowledge graphs


The implemtation of Match is in process.py

![example bob dylan](https://raw.githubusercontent.com/theblackcat102/language-models-are-knowledge-graphs-pytorch/main/images/bob_dylan.png)

### Execute MAMA(Match and Map) section



```
python extract.py examples/example2.txt bert-large-cased-example2.jsonl --language_model bert-large-cased --use_cuda true

python extract.py examples/example2.txt bert-large-cased-example2.jsonl --language_model bert-large-cased --use_cuda true

python extract.py examples/barack_obama.txt bert-large-cased-barack_obama.jsonl --language_model bert-large-cased --use_cuda true

python extract.py examples/bob_dylan.txt bert-large-cased-bob_dynlan.jsonl --language_model bert-large-cased --use_cuda true
```

## Map

1. Entity linking
   we had used Spacy Entity Linker
   Manually db has to be downloaded

   wget "https://wikidatafiles.nyc3.digitaloceanspaces.com/Hosting/Hosting/SpacyEntityLinker/datafiles.tar.gz" -O /tmp/knowledge_base.tar.gz
   tar -xzf /tmp/knowledge_base.tar.gz --directory ./data_spacy_entity_linker
   rm /tmp/knowledge_base.tar.gz

   if you get exception in map make sure path in anaconda3/lib/python3.8/site-packages/spacyEntityLinker/DatabaseConnection.py
   DB_DEFAULT_PATH = '../../data_spacy_entity_linker/wikidb_filtered.db'
   it should be as per data_spacy_entity_linker db 



2. Relations linking (page 5, 2.2.1)

Lemmatization is done in the previous steps [process.py](), in this stage we remove inflection, auxiliary verbs, adjectives, adverbs words.

Adjectives extracted from here: [https://gist.github.com/hugsy/8910dc78d208e40de42deb29e62df913](https://gist.github.com/hugsy/8910dc78d208e40de42deb29e62df913)


Adverbs extracted from here : [https://raw.githubusercontent.com/janester/mad_libs/master/List%20of%20Adverbs.txt](https://raw.githubusercontent.com/janester/mad_libs/master/List%20of%20Adverbs.txt)


### Environment setup


This repo is run using virtualenv 

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

