#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:27:37 2021

@author: pushpa
"""

#!/usr/bin/env python
# coding: utf-8

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
import os
import numpy as np
import pandas as pd
import re
from os import listdir
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from openie import StanfordOpenIE
os.environ["CORENLP_HOME"] = r'/home/pushpa/Documents/stanford-corenlp-latest/stanford-corenlp-4.2.0/'
from stanfordnlp.server import CoreNLPClient
import pandas as pd
import re
import spacy
import neuralcoref
nlp = spacy.load('/home/pushpa/.local/lib/python3.6/site-packages/en_core_web_lg/en_core_web_lg-2.1.0')

neuralcoref.add_to_pipe(nlp)


"""we have used custome stop words along with nltk stop words"""
def create_stop_words():
    with open("fluff.txt") as f:
        stop_words = list(set([line.strip() for line in f]))
    f.close()
    return stop_words
##########################################################################################
domain_list=[]
domain_dataset={}
def create_wiki_dataset():
    global domain_list
    global domain_dataset
    title = "wiki_data"
    folders = [x[0] for x in os.walk(str(os.getcwd())+'/'+title+'/')]   
    print("folder",folders)
    for folder in folders[1:]:
        file = open(folder+"/Master_PageTitles.txt", 'w')
        for f in listdir(folder):
            file.write(os.path.splitext(f)[0])
            file.write("\n")
        file.close()
    domain_list=[x for x in folders[1:]]
    #Collecting the file names and titles
    file_name=[]
    file_title=[]
    for i in folders[1:]:
        dataset = []
        domain_name=(i.split(os.path.sep)[-1])
        file = open(i+"/Master_PageTitles.txt", 'r')
        text = file.read().strip()
        file.close()    
        file_name =[x+".txt" for x in text.splitlines( )]
        file_title =[x for x in text.splitlines( )]                
        print(len(file_name), len(file_title))
        for j in range(len(file_name)):
            dataset.append((str(i) +"/"+ str(file_name[j]), file_title[j]))
        domain_dataset[domain_name]=dataset

###############################################################################
#below are functions to preprocess text dataset created from wikipedia
def lemmatization(data):    
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    tokens = word_tokenize(str(data))
    lemma_function = WordNetLemmatizer()
    new_text = ""
    for token, tag in pos_tag(tokens):
        lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
        new_text = new_text+" "+lemma
    return new_text.strip()



def convert_lower_case(data):
    return np.char.lower(str(data))

def remove_stop_words(data):
    nltk_stop_words = stopwords.words('english')
    stop_words=create_stop_words()
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in nltk_stop_words and len(w) > 4:
            if w not in stop_words:
                new_text = new_text + " " + w
    return new_text.strip()


def remove_punctuation(data):
   
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n=="
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text.strip()


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text



def remove_number(data):
    string_no_numbers = re.sub("\d+", " ", str(data))
    return string_no_numbers 


##############################################################################

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    #data = remove_stop_words(data)
    data=remove_number(data)
    data = convert_numbers(data)
    data = lemmatization(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = lemmatization(data) #needed again as we need to stem the words
    data=remove_number(data)
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data



##############################################################################
#using preprocessing functions we are creating clean domain data
def GetCleanDomainText(domain_name):
    processed_text = []
    processed_title = []

    for i in domain_dataset[domain_name]:
        print(i[0])
        try :
            file  = open(i[0], 'r', encoding="utf8", errors='ignore')
            text = file.read().strip()
            file.close()
        except  FileNotFoundError:
              print("File does not exist")
        

        processed_text.append(text)
        processed_title.append(i[1])
        
    return processed_text,processed_title
   


##############################################################################
def CreateREL(domain_name):
    global domain_dataset
    triples_corpus_dataframe = pd.DataFrame()
    print(domain_dataset.keys())
    with StanfordOpenIE() as client:
        for i in domain_dataset[domain_name]:
            #print(i[0])
            text = " "
            try :
                file  = open(i[0], 'r', encoding="utf8", errors='ignore')
                text = file.read().strip()
                file.close()
            except  FileNotFoundError:
                  print("File does not exist")
            text = text.replace('\n', ' ').replace('\r', '')
            triples_corpus = client.annotate(text[0:70000], properties={"annotators":"coref",'pipelineLanguage': 'en',
                                 'coref.algorithm' : 'statistical',                                        
                                "outputFormat": "json",
                                "splitter.disable": "true",
                                 "openie.triple.strict":"true",
                                 "openie.max_entailments_per_clause":"1"})
            #print('Corpus: %s [...].' % text[0:80])
            #print('Found %s triples in the corpus.' % len(triples_corpus),triples_corpus[0])
            triples_corpus_dataframe = triples_corpus_dataframe.append(triples_corpus, ignore_index=True)
        triples_corpus_dataframe.to_csv(domain_name+".csv")
    return 1
############################################################################## 
def CreateREL_Coref(domain_name):
    global domain_dataset
    triples_corpus_dataframe = pd.DataFrame()
    print(domain_dataset.keys())
      
    for i in domain_dataset[domain_name]:
        #print(i[0])
        text = " "
        try :
            file  = open(i[0], 'r', encoding="utf8", errors='ignore')
            text = file.read().strip()
            file.close()
        except  FileNotFoundError:
              print("File does not exist")
        text = text.replace('\n', ' ').replace('\r', '')
        # submit the request to the server
        client = CoreNLPClient(properties={'annotators': 'coref', 'coref.algorithm' : 'statistical'}, timeout=60000, memory='16G')
        ann = client.annotate(text)    
        
        mychains = list()
        chains = ann.corefChain
        for chain in chains:
            mychain = list()
            # Loop through every mention of this chain
            for mention in chain.mention:
                # Get the sentence in which this mention is located, and get the words which are part of this mention
                # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
                words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
                #build a string out of the words of this mention
                ment_word = ' '.join([x.word for x in words_list])
                mychain.append(ment_word)
                mychains.append(mychain)
        #print('Corpus: %s [...].' % text[0:80])
        #print('Found %s triples in the corpus.' % len(triples_corpus),triples_corpus[0])
            triples_corpus_dataframe = triples_corpus_dataframe.append(mychains, ignore_index=True)
        triples_corpus_dataframe.to_csv(domain_name+"coref.csv")
        return 1
##############################################################################
def get_entity_pairs(text, coref=True):
    # preprocess text
    text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
    text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
    text = nlp(text)
    if coref:
        text = nlp(text._.coref_resolved)  # resolve coreference clusters

    def refine_ent(ent, sent):
        unwanted_tokens = (
            'PRON',  # pronouns
            'PART',  # particle
            'DET',  # determiner
            'SCONJ',  # subordinating conjunction
            'PUNCT',  # punctuation
            'SYM',  # symbol
            'X',  # other
        )
        ent_type = ent.ent_type_  # get entity type
        if ent_type == '':
            ent_type = 'NOUN_CHUNK'
            ent = ' '.join(str(t.text) for t in
                           nlp(str(ent)) if t.pos_
                           not in unwanted_tokens and t.is_stop == False)
        elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
            refined = ''
            for i in range(len(sent) - ent.i):
                if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                    refined += ' ' + str(ent.nbor(i))
                else:
                    ent = refined.strip()
                    break

        return ent, ent_type

    sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
    ent_pairs = []
    for sent in sentences:
        sent = nlp(sent)
        spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
        #spans = spacy.util.filter_spans(spans)
        spans = filter_spans(spans)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span, attrs={'tag': span.root.tag,
                                            'dep': span.root.dep}) for span in spans]
        deps = [token.dep_ for token in sent]

        # limit our example to simple sentences with one subject and object
        if (deps.count('obj') + deps.count('dobj')) != 1\
                or (deps.count('subj') + deps.count('nsubj')) != 1:
            continue

        for token in sent:
            if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
                continue
            subject = [w for w in token.head.lefts if w.dep_
                       in ('subj', 'nsubj')]  # identify subject nodes
            if subject:
                subject = subject[0]
                # identify relationship by root dependency
                relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                if relation:
                    relation = relation[0]
                    # add adposition or particle to relationship
                    if relation.nbor(1).pos_ in ('ADP', 'PART'):
                        relation = ' '.join((str(relation), str(relation.nbor(1))))
                else:
                    relation = 'unknown'

                subject, subject_type = refine_ent(subject, sent)
                token, object_type = refine_ent(token, sent)

                ent_pairs.append([str(subject), str(relation), str(token),
                                  str(subject_type), str(object_type)])

    ent_pairs = [sublist for sublist in ent_pairs
                          if not any(str(ent) == '' for ent in sublist)]
    pairs = pd.DataFrame(ent_pairs, columns=['subject', 'relation', 'object',
                                             'subject_type', 'object_type'])
    print('Entity pairs extracted:', str(len(ent_pairs)))

    return pairs

#######################################################################################
def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result

##########################################################################################

              
# create wiki datasets from folders 
create_wiki_dataset()
total_documents=0
all_domain_token=[]
domain_name ="Film"
processed_text,processed_title= GetCleanDomainText(domain_name)
processed_text_dataframe = pd.DataFrame(processed_text)
processed_text_dataframe.to_csv("processed_text_dataframe.csv")
wiki_data = pd.read_csv("processed_text_dataframe.csv")
wiki_data.columns=["No","text"]
#CreateREL("Film")
   
#CreateREL_Coref("Film")    

