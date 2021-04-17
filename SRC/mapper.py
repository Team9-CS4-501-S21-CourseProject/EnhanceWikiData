from constant import invalid_relations_set
KG_results = []
from REL.db.generic import GenericLookup
import spacy
from spacyEntityLinker import EntityLinker

#Initialize Entity Linker
entityLinker = EntityLinker()

#initialize language model
nlp = spacy.load("en_core_web_sm")
#add pipeline
nlp.add_pipe(entityLinker, last=True, name="entityLinker")

Genric_Folder="/home/pushpa/Documents/shared_folder/Project_SRC_Results/SourceFiles/generic/"
adjectives_list_file = Genric_Folder+'english-adjectives.txt'
adverbs_list_file = Genric_Folder+'adverbs.txt'

#load invalid relations from file
with open(adjectives_list_file, 'r') as f:
    adjectives = [ line.strip().lower() for line in f]

with open(adverbs_list_file, 'r') as f:
    adverbs = [ line.strip().lower() for line in f]

lexically_invalid_relations = [
    'and', 'but', 'or', 'so', 'because', 'when', 'before', 'although', # conjunction
    'oh', 'wow', 'ouch', 'ah', 'oops',
    'what', 'how', 'where', 'when', 'who', 'whom',
    'a', 'and', 'the', 'there', 
    'them', 'he', 'she', 'him', 'her', 'it', # pronoun
    'ten', 'hundred', 'thousand', 'million', 'billion',# unit
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',# number
    'year', 'month', 'day', 'daily',
]

auxiliaries = [
    'be', 'can', 'have', 'dare', 'may', 'will', 'would', 'should', 
    'need', 'ought', 'shall', 'might', 'do', 'does', 'did',
    'be able to', 'had better','have to','need to','ought to','used to',
]
heuristically_invalid = [
    'and', 'of', 'in', 'to', ',', 'for', 'be', 'by', 'with', 'on', 'as', 'that', 'from', 'be', ')', '(', 'which',
    'at', 'be', 'be', 'be', ';', 'or', 'but', 'have', 'have', 'the', 'have', 'not', 'after', '"', 'include', 'also',
    'be', 'into', 'between', 'such', ':', 'do', 'while', 'when', 'during', 'would', 'over', 'since', '2019', 
    'well', 'than', '2020', 'under', 'where', 'one', 'be', 'hold', '2018', 'can', 'through', '-', 
    'make',  'out', 'there', 'know', 'due', 'a', 'take', 'up', 'begin', 'before', 'about',
    "'",  '4', '10', '3', '11', '&', '$', '12',  '2015', '2008','–', 'will',
    'so', 'do', 'follow', 'most', 'although', 'cause', 'only', '—',  '2007',  '2014', 'mostly', '5', 'say', '2017', '20', 
    '2009',
]

def getInvalidRelationsList():
  invalidRelations = lexically_invalid_relations
  invalidRelations += auxiliaries
  invalidRelations += heuristically_invalid
  invalidRelations += adjectives
  invalidRelations += adverbs

  return set(invalidRelations)


invalid_relations_set = getInvalidRelationsList()    


def getSpacyEntityResolved(iptext):
  
    try:
        doc = nlp(iptext)
    #returns all entities in the whole document
        all_linked_entities=doc._.linkedEntities
        test_entity=doc._.linkedEntities[0]
        test_entity.pretty_print() 
        return test_entity.get_id()  
    except Exception:
        print(Exception)
        return None

def Map(head, relations, tail,conf, top_first=True, best_scores = True):
    global invalid_relations_set
    kg_head = ''
    kg_relation = relations
    kg_tail = ''
    kg_entry_type = 'KG'
    kg_newEntity = ''
    kg_confidenceScore = conf
    if head == None or tail == None or relations == None:
        return {}
    #print(head,tail,relations)
    if relations[0] not in invalid_relations_set:
        head_p_e_m = getSpacyEntityResolved(head)
        if head_p_e_m is None:
            kg_entry_type = 'Open_KG'
            kg_head = head
            kg_newEntity += 'head_'
        else :
            kg_head = 'Q'+str(head_p_e_m)

        tail_p_e_m = getSpacyEntityResolved(tail)
        if tail_p_e_m is None:
            kg_entry_type = 'Open_KG'
            kg_tail = tail
            kg_newEntity += 'tail_'
        else :
            kg_tail = 'Q'+str(tail_p_e_m)
        openGraphEntry = { 'h': kg_head, 't': kg_tail, 'r': kg_relation,'type':kg_entry_type,'newEntities':kg_newEntity ,'pair_c':kg_confidenceScore }
        KG_results.append(openGraphEntry)
    else :
      print('Found Invalid Relation')    
    print(kg_head,kg_tail,relations)
    return { 'h': kg_head, 't': kg_tail, 'r': '_'.join(relations)},KG_results

def deduplication(triplets):
    unique_pairs = []
    pair_confidence = []
    for t in triplets:
        key = '{}\t{}\t{}'.format(t['h'], t['r'], t['t'])
        conf = t['c']
        if key not in unique_pairs:
            unique_pairs.append(key)
            pair_confidence.append(conf)
    
    unique_triplets = []
    for idx, unique_pair in enumerate(unique_pairs):
        h, r, t = unique_pair.split('\t')
        unique_triplets.append({ 'h': h, 'r': r, 't': t , 'c': pair_confidence[idx]})

    return unique_triplets



