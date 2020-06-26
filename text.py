import spacy
import textacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy import displacy
from itertools import product
from functools import reduce, partial
from operator import add

import en_core_web_lg

get_text = lambda x: x[0].doc[ x[0].i: x[-1].i + 1].text
get_doc = lambda x: x[0].doc[ x[0].i: x[-1].i + 1]
ent_types =  {'PERSON','NORP','GPE','ORG','FAC','LOC','EVENT','PRODUCT'}

def should_recur(toks) -> bool :
    verb = len([x.pos_ for x in get_doc(toks) if x.pos_ in {'VERB'} ]) > 0
    ent = len(get_doc(toks).ents) > 0
    return verb and ent 

def should_keep(potential) -> bool : 
    try:
        left, verb, right = potential
        right = tuple(filter(lambda x: x not in verb , get_doc(right)))
        left =  tuple(filter(lambda x: x not in verb , get_doc(left)))
        #check for only punctuation e's 
        if (len(left) == 1 and not left[0].text.isalnum()) : return False 
        if (len(right) == 1 and not right[0].text.isalnum()) : return False 
        e = ( len([x for x in get_doc(left).ents if x.label_ in ent_types  ]) +len([x for x in get_doc(right).ents if x.label_ in ent_types ])) > 0
        nov = (len([x.pos_ for x in get_doc(right) if x.pos_ in {'VERB'} ]) + len([x.pos_ for x in get_doc(left) if x.pos_ in {'VERB'} ])  ) == 0
    except Exception as e:
        #print(potential)
        return False
    return e and nov

def get_potentials(nlp, matcher, text:str) -> list:

    if len(text.split(' ')) <= 1:
        return [] 

    potentials = []
    #process the sentence, and search for verb phrases
    doc = nlp(text)

    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    verb_phrases = filter_spans(spans)
    
    for verb in verb_phrases:
        #for each verb phrase, get the rights and lefts. 
        #build all potential combinations 

        lefts = verb.root.lefts
        lefts = [tuple(x.subtree) for x in lefts]

        rights = verb.root.rights
        rights = [tuple(x.subtree) for x in rights]

        potentials.extend([(a,verb,b) for a,b in product(lefts,rights)])

    #filter potentials - Only ones that either have VERBS in them or ENTITIES in them 

    for potential in potentials: 
        left, verb, right = potential

        lpotents = get_potentials(nlp, matcher, get_text(left)) if should_recur(left) else [] 
        rpotents = get_potentials(nlp, matcher, get_text(right)) if should_recur(right) else [] 

        potentials.extend(rpotents)
        potentials.extend(lpotents)

    return potentials


def get_unique_relationships(relationships):
    unique = set()
    for e1, v, e2 in relationships:
        e1 = tuple(filter(lambda x: x not in v , get_doc(e1)))
        e2 =  tuple(filter(lambda x: x not in v , get_doc(e2)))
        key = '|'.join(map(get_text, [e1,v,e2])) + '|' + v[0].sent.text
        unique.add(key)
    return list(unique)


def extract_relationships(nlp, matcher, text) :
    """Recursively extract relationships from text by considering verbs.
    Params: 
        - nlp - the Spacy model used to perform NER. 
        - matcher - the spacy matcher object used to find verb phrases.
        - text - the text to analyze with the spacy model 
        - 
    Returns :
        - results - extracted relationships in the format of : [entity 1 ] [verb phrase] [entity 2]
    """
    #process the sentence, and search for verb phrases
    relationships = get_potentials(nlp, matcher, text)
    relationships = list( filter (should_keep, relationships))
    relationships = get_unique_relationships(relationships)
    return relationships

####example use case

nlp = en_core_web_lg.load()

texts ="""The US government must identify and prioritize actual or potential terrorist sanctuaries. For each, it should have a realistic strategy to keep 
possible terrorists insecure and on the run, using all elements of national power. We should reach out, listen to, and work with other countries that can help. 
Our report shows that al Queda has tried to acquire or make weapons of mass destruction for at least 10 years. There is no doubt the United States would be the 
prime target. 
Targeting travel is at l east as powerful a weapon against terrorists as targeting their money. The United States should combine terrorist travel intelligence, 
operations, and law enforcement in a strategy to intercept terrorists, find terrorist facilitators, and constrain terrorist mobility. 
We recommend the establishment of a National Counterterrorism Center (NCTC) built on the foundation of the existing TTIC. Breaking the older mold of national 
government organization, this National Counterterrorism Center should be a center for join operation planning and joint intelligence, staffed by personnel from the 
various agencies. The head of the National Counterterrorism Center should have the authority to evaluate the performance 
of the people assigned to the Center. The President determines the guidelines for information sharing among government agencies and by those agencies with the private sector, he should safeguard 
the privacy of individuals about whom the information is shared. 
Bill sent out the consulting agreement to the rest of Shopify""".replace('\n','')

texts = texts.split('. ')


pattern = [
    {'POS': 'VERB', 'OP': '?'},
    {'POS': {'IN': ['ADV','AUX','CCONJ','PART']}, 'OP':'*'},
    {'POS': 'VERB', 'OP': '+'},
    {'POS': {'IN': ['ADV','AUX','CCONJ','PART']}, 'OP':'*'},
    {'POS': 'VERB', 'OP': '?'}

]

# instantiate a Matcher instance
matcher = Matcher(nlp.vocab)
matcher.add("Verb phrase", None, pattern)
#curry function
er = partial(extract_relationships, nlp, matcher)

results = list(map(er, texts))
results = reduce(add, results)
print(results)
