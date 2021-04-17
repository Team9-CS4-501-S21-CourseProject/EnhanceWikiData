import pandas as pd
import sys, os
from process import parse_sentence
from mapper import Map, deduplication
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import en_core_web_md
from tqdm import tqdm
import json
import os
import torch
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Process lines of text corpus into knowledgraph')
parser.add_argument('input_filename', type=str, help='text file as input')
#parser.add_argument('output_filename', type=str, help='output text file')
parser.add_argument('--language_model',default='bert-base-cased', 
                    choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                    help='which language model to use')
parser.add_argument('--use_cuda', default=True, 
                        type=str2bool, nargs='?',
                        help="Use cuda?")
parser.add_argument('--include_text_output', default=False, 
                        type=str2bool, nargs='?',
                        help="Include original sentence in output")
parser.add_argument('--threshold', default=0.05, 
                        type=float, help="Any attention score lower than this is removed")

args = parser.parse_args()

use_cuda = args.use_cuda
nlp = en_core_web_md.load()
output_folder="/home/pushpa/Documents/shared_folder/Project_SRC_Results/language-models-are-knowledge-graphs-pytorch-main/output_film_with_tol_0.5/"
'''Create
Tested language model:

1. bert-base-cased

2. gpt2-medium

Basically any model that belongs to this family should work

'''

language_model = args.language_model



input_filename = args.input_filename
wiki_data = pd.read_csv(input_filename)
wiki_data.columns=["drop","text","drop"]
coref_resol=pd.DataFrame(columns=["coref_text"])
wiki_data= wiki_data[wiki_data["text"]!="None"]





if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    if 'gpt2' in language_model:
        encoder = GPT2Model.from_pretrained(language_model)
    else:
        encoder = BertModel.from_pretrained(language_model)
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()    
    input_filename = args.input_filename
    output_filename = input_filename+"out.json"
    include_sentence = args.include_text_output

    for index, row in wiki_data.iterrows():
	    with open(output_folder+"coref_resol_"+str(index)+"_out.json", 'w') as g:
	            sentence  = str(row["text"]).strip()
	            if len(sentence):
	                valid_triplets = []
	                for sent in nlp(sentence).sents:
	                    # Match
	                    for triplets in parse_sentence(sent.text, tokenizer, encoder, nlp, use_cuda=use_cuda):
	                        valid_triplets.append(triplets)
	                if len(valid_triplets) > 0:
	                    # Map
	                    mapped_triplets = []
	                    for triplet in valid_triplets:
	                        head = triplet['h']
	                        tail = triplet['t']
	                        relations = triplet['r']
	                        conf = triplet['c']
	                        #print("head",head,"tails",tail,"relations",relations)
	                        if conf < args.threshold:
	                            continue
	                        mapped_triplet = Map(head, relations, tail)
	                        if 'h' in mapped_triplet:
	                            mapped_triplet['c'] = conf
	                            mapped_triplets.append(mapped_triplet)
	                            #print(mapped_triplet)
	                    output = { 'line': index, 'tri': deduplication(mapped_triplets) }
	                    if include_sentence:
	                        output['sent'] = sentence
	                    if len(output['tri']) > 0:
	                        g.write(json.dumps( output )+'\n')
	                        g.close()
	                        gc.collect()
	                        torch.cuda.empty_cache()

