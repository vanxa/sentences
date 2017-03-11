import nltk
from nltk.corpus import wordnet as wn

def get_synonyms(word, pos=None):
    syns = []
    try:
        for syn_set in wn.synsets(word,pos):\
            syns += syn_set.lemma_names
    except KeyError:
        print "Invalid POS tag! Getting all synonyms.."
        return get_synonyms(word)
    return set(syns)
