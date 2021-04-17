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
    data = remove_stop_words(data)
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
        

        processed_text.append(word_tokenize(str(preprocess(text))))
        processed_title.append(word_tokenize(str(preprocess(i[1]))))
        
    return processed_text,processed_title
   


##############################################################################
def CreateREL(domain_name):
    global domain_dataset
    print(domain_dataset.keys())
    with StanfordOpenIE() as client:
        for i in domain_dataset[domain_name]:
            print(i[0])
            text = " "
            try :
                file  = open(i[0], 'r', encoding="utf8", errors='ignore')
                text = file.read().strip()
                file.close()
            except  FileNotFoundError:
                  print("File does not exist")
            text = text.replace('\n', ' ').replace('\r', '')
            triples_corpus = client.annotate(text[0:50000])
            print('Corpus: %s [...].' % text[0:80])
            print('Found %s triples in the corpus.' % len(triples_corpus))
            for triple in triples_corpus[:3]:
                print('|-', triple)
    return 1
##############################################################################               
# create wiki datasets from folders 
create_wiki_dataset()
total_documents=0
all_domain_token=[]


                
                
CreateREL("Film")
   
    

