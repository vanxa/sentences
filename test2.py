import nltk
from nltk.corpus import wordnet as wn

THRESH = 0.1

def find_similar_synsets(word1,word2):

    synset1 = wn.synsets(word1)
    synset2 = wn.synsets(word2)
    most_similar = []
    lst = [(syn1,syn2) for syn1 in synset1 for syn2 in synset2]
    res = {}
    for syn1,syn2 in lst:
        sim = syn1.path_similarity(syn2)
        if sim > THRESH:
            res[syn1.name+"-"+syn2.name] = sim
    return res
    
