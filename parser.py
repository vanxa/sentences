# author: Ivan Konstantinov
# This is the script to create complex sentence trees and cooresponding cpfg grammars, given a training data file and an existing grammar
# To create a new grammar, the simplest way is to import this file and run 'parser.run_grammar()'. This will create a new grammar file,
# 'nltk_grammar.cfg'. 
# To use the parser, run 'parser.run_parser()'. Additionally, you can run everything in one go by issuing 'parser.run_all()'.
# Currently, the parser has not been tested beyound the get_trees() function, as it is apparently a lengthy process. I'll provide some debugging
# and will also try to minimze the grammar file, as it is quite big at the moment. One way to do this is to manually select which Treebank corpus
# sections to use when deriving the grammar, instead of using the whole corpus.

import os
import nltk
import re
from nltk.tag.simplify import simplify_wsj_tag as simplify
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import random
from sys import stdout
from itertools import combinations 

ACCESS_APPEND="a"
ACCESS_OVERWRITE="w"


class MyParser:
    def main(self):
        print "Starting grammar...\nTokenizing the input file..."
        result = []
        if len(self.grammar.train) > 1:
            print str(len(self.grammar.train)) + " sentences found in input file...."
            for sentence in self.grammar.train:
                print "Parsing..."
                parsed_sentence = self.parse_sentence(sentence)
                result += [parsed_sentence]
        else:
            parsed_sentence = self.parse_sentence(self.grammar.train[0])
            result = [parsed_sentence]
        print "The following synonymized sentences were created:\n"
        for sent in result:
            print sent
        return result

    def __init__(self, filename):
        # the data
        print "Initializing new parser..."
        print "Initializing grammar..."
        self.grammar = MyGrammar(filename, standalone = False)
        print "Done."
        print "Creating synonym index..."
        self.index = self.create_synindex(self.grammar.inv_index)
        print "Done."

    def parse_sentence(self,sentence):
        print "Original sentence: " + sentence
        tokens = self.grammar.tokenize([sentence]) # Train is not a very intuitive name here
        synonymized_txt = self.parse_synonyms(tokens)
        parsed = ""
        words = [words for words, tags in tokens]
        for word in words:
            if word != synonymized_txt[word]:
                parsed += synonymized_txt[word].upper() + " "
            else:
                parsed += word + " "
        return parsed 


    def parse_synonyms(self,tokens):
        synonymized_txt = {}
        inv_index = self.grammar.inv_index
        for word,tag in tokens:
            if word in inv_index:
                groups = [group for group in inv_index[word] if tag in group]
                if len(groups) == 0:
                    print "There are no synonyms for \""+word+"\" that are of pos tag \""+tag+"\""
                    synonymized_txt[word] = word
                    continue

                try:
                    while True:
                        rand_group = random.randint(0,len(groups)-1)
                        group = groups[rand_group]
                        syns = self.index[group]
                        if len(group) > 1 and len(syns) > 1:
                            break;

                    while True:
                        rand_word = random.randint(0,len(syns)-1)
                        c_word = syns[rand_word]
                        if c_word != word:
                            print "Found a synonym for \""+word+"\":\""+c_word+"\""
                            break
                    synonymized_txt[word] = c_word

                except ValueError:
                    print "ERROR " + groups
            else:
                synonymized_txt[word] = word
        return synonymized_txt


    
    # DEPRECATED
    def get_grammar(self):
        self.grammar = nltk.data.load("file:"+self.grammar_cfg)
        self.chart_parser = nltk.ChartParser(self.grammar)

    # DEPRECATED
    def get_trees(self):
        #for line in self.data:
            line = self.data[0]
            print 'Getting sentence trees for sentence:\n ' + line
            try:
                self.trees += self.chart_parser.chart_parse(line.split(),1)
            except Exception:
                print 'line: ' + line

    # DEPRECATED
    def get_productions(self):
        for tree in self.trees:
            self.productions += tree.productions()

    # DEPRECATED
    def create_pcfg(self):
        self.pcfg = nltk.induce_pcfg("S",self.productions)

    # Creates an index file from the provided inverted index, that is, derives the [tag] => [words] relationship
    def create_synindex(self,inv_index):
        index = {}
        for word in inv_index:
            for group in inv_index[word]:
                if group in index and word not in index[group]:
                    index[group] += [word]
                elif group not in index:
                    index[group] = [word]
        return index



################################### START GRAMMAR ################################
# A grammar class extracts grammar rules (nonterminal) from the Treebank corpus
# and creates terminal rules given a training file
class MyGrammar:

    # stores the train data
    global train 
    # The nonterminal rules that form the basis of the grammar 
    global nont_rules 
    # the number of synonym groups
    global syngroups
    # word rules stored in this file are used to create the resulting grammar
    global word_rules
    # the inverted index used to store the synonym group for each word
    global inv_index
    # defines the threshold beyond which two words are similar, using the synset.path_similarity() method
    global thresh 
    # stores the pos tag for each word
    global pos_tags
    # the tagger to be used
    global tagger

    # Filenames
    index_fl = "inv_index.dat"
    rules_fl = "rules.dat"
    gramm_fl = "grammar.dat"
    nont_fl = "core.dat"
    syngroups_fl = "syngroups.dat"
    # the new word rules to be appended to the existing ones
    rules_out = ""
    # the resulting grammar rules to be written to file
    grammar_out = ""
    # the list of syngroups to be appended to the existing ones
    syngroups_out = ""
    # the new index entries to be appended
    index_out = ""

    def __init__(self, filename, standalone = True):
        print "Reading train data from " + filename
        if standalone:
            self.train= pre_process(read_file(filename)).split('\n')[:-1]
        else:
            self.train = read_file(filename).split('\n')[:-1]
        print "Reading word rules"
        self.word_rules = self.get_word_rules()
        print "Reading synonym group inverted index"
        self.inv_index = self.get_inv_index()
        print "Getting synonym groups"
        self.syngroups = self.get_syngroups()
        print "Getting grammar"
        self.nont_rules= self.get_nont_rules()
        if standalone:
            print "Initializing taggers"
            rand = random.randint(20,1000)
            print "Random train seed: " + str(rand)
            brown_train = brown.tagged_sents(simplify_tags=True)[rand:]
            rxtagger = nltk.RegexpTagger([(r'.*','NN')])
            utagger = nltk.UnigramTagger(brown_train,backoff=rxtagger)
            self.tagger = nltk.BigramTagger(brown_train,backoff=utagger)
            print "Done."
        
        self.grammar_out = ""
        # Pick a random threshold
        self.thresh = random.uniform(0.35,0.5)
        print "Similarity Threshold for this session: " + str(self.thresh)


    # runs the whole program.
    def build_grammar(self):
        print "Building grammar.....\n"
        print 'Done.\nTokenizing train data...'
        tags = self.tag(self.train)
        self.set_pos_tags(tags)
        print 'Building synonym list for each token'
        syns = self.get_synonyms_v2(tags)
        return syns
        print "Populating grammar rules and inverted list"
        self.process_synonyms(syns)

        print 'Done.\nWriting to file...'
        self.create_grammar()
        print 'Done.'


    # new train data
    def load_new_content(self,filename):
        self.train = pre_process(read_file(filename))

    # new train data, no pre-processing
    def load_new_content_raw(self,filename):
        self.train = read_file(filename)

	# DEPRECATED
    # run a parser throught the train data
    def tokenize(self, data):
        tags = []
        print 'Got ' + str(len(data)) + ' lines of train data.'
        for line in data:
           stdout.write(".")
           stdout.flush()
           tags += self.tag_text(line)
        return tags
    
    def tag(self, train):
    	print "Got " + str(len(train)) + " lines of train sentences. Flattening structure..."
    	train_joined = [word for sentence in train for word in sentence.split()]
    	print "Done. Getting tags..."
    	return self.filter_nonsig(self.tagger.tag(train_joined))
    	
    def filter_nonsig(self,tokens):
        nonsignificants = ["cnj","to","wh","ex","mod"]
        filtered = []
        for word,tag in set(tokens):
            if tag.lower() in nonsignificants:
                print "Filtering out nonsignificant \""+word+"\".. with tag \""+tag+"\""
            else:
                filtered += [(word,tag)]
        return filtered

    # tagger
    def tag_text(self,line):
        tags =  nltk.pos_tag(nltk.word_tokenize(line))
        return [(word,simplify(tag)) for word,tag in tags]

######## get_synonyms v.1 #############
    def get_synonyms(self,tags):
        syns = {} 
        for word, tag in tags:
            try:
                syns[word] += self.parse_synonyms(word,tag)
            except KeyError:
                syns[word] = self.parse_synonyms(word,tag)
        return syns

    def parse_synonyms(self,word,tag=None):
        syns = []
        try:
            #for syn_set in wn.synsets(word,tag):
            #    syns += syn_set.lemma_names
            synsets = wn.synsets(word,tag)
            if len(synsets) == 1:
                print "Word \"" + word +"\" has only one synset"
                syns += synsets[0].lemma_names
        #    elif len(synsets) > 1:
        except KeyError:
            print "Invalid pos tag! Getting all synonyms for word " + word
            return self.parse_synonyms(word)
        return list(set(syns))
######## END get_synonyms v.1 ###########

####### get_synonyms v.2 ##############

###### END get_synonyms v.2 ##########
    # The difference in v.2 is that it takes only the best computed matches, based on the rest of the input file
    def get_synonyms_v2(self, tags):
        new_words = [(word,tag) for word,tag in tags if word not in self.inv_index] 
        synsets = {}
        synonyms = {}
        words = []
        for word,tag in new_words:
            words += [word]
            pos = None
            if tag == "N":
                pos = wn.NOUN
            elif tag == "V":
                pos = wn.VERB
            elif tag == "ADJ":
                pos = wn.ADJ
            elif tag == "ADV":
                pos = wn.ADV
            # the list for each synset contains the synonyms within the text 
            print "Getting the synsets for \"" + word +"\""
            ssets = wn.synsets(word,pos)
            for sset in ssets:
                if sset.pos == pos:
                    try:
                        synsets[sset] += [word]
                    except KeyError:
                        synsets[sset] = [word]

        print "Creating synset pairs"
        # Very slow operation!
        pairs = self.pairwise(synsets.keys())
        # compute the similarity between each pair of synsets
        print "Checking pairs"
        for syn1, syn2 in pairs:
            if syn1 != syn2 and syn1.pos == syn2.pos:
                sim = syn1.path_similarity(syn2)
                if sim >= self.thresh:
                    # these synsets are similar
                    key = synsets[syn1][0]
                    tag = syn1.pos
                    if key in synonyms:
                        try:
                            synonyms[key][tag] = list(set(synonyms[key][tag] + synsets[syn1] + synsets[syn2]))
                        except KeyError:
                            synonyms[key][tag] = list(set(synsets[syn1] + synsets[syn2]))
                    else:
                        synonyms[key] = { tag : list(set(synsets[syn1] + synsets[syn2])) }

        syn_words = [ word for sublist in [ synsets[key] for key in synsets ] for word in sublist]
        # find the difference between the original list of words and the list of "good" synonyms. 
        diff = self.difference(words, syn_words)
        for word in diff:
            print "Word \"" + word + "\" does not have a \'good\' synonym"
            # Take the first two synsets for those words that do not appear in the "good" synonym list
            word_synsets = wn.synsets(word)
            if len(word_synsets) == 0:
                synonyms[word] = []
            elif len(word_synsets) > 1:
                synonyms[word] = list(set(word_synsets[0].lemma_names + word_synsets[1].lemma_names))
            else:
                synonyms[word] = list(set(word_synsets[0].lemma_names))
        # done
        return synonyms


    def pairwise(self,iterable):
        pairs = combinations(iterable,2) 
        return pairs

    def difference(self, list1, list2):
        return list(set(list1) - set(list2))

    def get_word_rules(self):
        rules = {}
        content = read_file(self.rules_fl)
        if content != '':
            for line in content.split('\r\n')[:-1]:
                try:
                    word,tag = line.split(":")
                    rules[word] = tag
                except ValueError:
                    print word
        return rules

    def get_inv_index(self):
        index = {}
        syngroups = []
        content = read_file(self.index_fl)
        if content != '':
            for line in content.split("\r\n")[:-1]:
                word,syngroup = line.split(":")
                syngroups += [syngroup]
                try:
                    index[word] += [syngroup]
                except KeyError:
                    index[word] = [syngroup]
        return index


    def get_nont_rules(self):
        return read_file(self.nont_fl)

    def get_syngroups(self):
        syngroups = {}
        data = read_file(self.syngroups_fl)
        if data != '':
            for line in data.split('\r\n')[:-1]:
                parent,child = line.split(":")
                try:
                    syngroups[parent] += [child]
                except KeyError:
                    syngroups[parent] = [child]
        return syngroups

    
    def process_synonyms(self,synset):
        for word in synset:
            syns = synset[word]
            tag = self.pos_tags[word] 
            print "The POS tag for \"" + word + "\" is \"" + tag + "\""
            if syns == []:
                print "No synonyms found for word \"" + word + "\". Checking rules..."
                if word not in self.word_rules:
                    print "Adding new rule: " + word + ":" + tag
                    self.rules_out += word+":"+tag+"\r\n"
                    self.word_rules[word] = tag
            else:
                try:
                    print "Word is in groups: " + str(self.inv_index[word]) + " . Putting all synonyms in groups"
                    for syn in synset[word]:
                        for group in self.inv_index[word]:
                            if syn in self.inv_index and group not in self.inv_index[syn]:
                                self.inv_index[syn] += [group]
                            elif syn not in self.inv_index:
                                print "\""+syn+"\" is not in inverted index. Adding..."
                                self.inv_index[syn] = self.inv_index[word]
                                break
                except KeyError:
                    print "Word \"" + word + "\" does not belong to any synonym group. Checking synonyms"
                    done = 0
                    for syn in syns:
                        print "Synonym of \"" + word + "\" is \"" + syn +"\""
                        try:
                            groups = self.inv_index[syn] 
                            print "Adding \"" + word + "\" to synonym groups " + str(groups)
                            self.inv_index[word] = groups
                            done = 1
                        except KeyError:
                            pass
                    if done == 0:
                        print "No compatible synonym groups were found. Creating a new group"
                        children = []
                        try:
                            children = self.syngroups[tag]
                        except KeyError:
                            print "Creating new syngroup of type " + tag
                            self.syngroups[tag] = []
                        newGroup = tag + str(len(children) + 1)
                        self.syngroups[tag] += [newGroup]
                        print "Name of new group is \"" + newGroup + "\". Populating group..."
                        self.inv_index[word] = [newGroup]
                        self.index_out += word + ":" + newGroup + "\r\n"
                        self.syngroups_out += tag + ":" + newGroup + "\r\n"
                        for syn in syns:
                            self.inv_index[syn] = [newGroup]
                            self.index_out += syn + ":" + newGroup +"\r\n"
                        print "Done"

    def create_grammar(self):
        print "Creating final grammar."
        print "Writing core rules...."
        self.grammar_out += self.nont_rules + "\r\n"
        print "Writing word rules..."
        for word in self.word_rules:
            self.grammar_out += self.word_rules[word] + " -> \"" + word + "\"\r\n"
        print "Writing synonym rules..."
        for tag in self.syngroups:
            for group in self.syngroups[tag]:
                self.grammar_out += tag + " -> " + group + "\r\n"
        for word in self.inv_index:
            for syngroup in set(self.inv_index[word]):
                self.grammar_out += syngroup + " -> \"" + word + "\"\r\n"
        print "Writing to files..."
        write_file(self.grammar_out, self.gramm_fl, ACCESS_OVERWRITE)
        write_file(self.rules_out, self.rules_fl, ACCESS_APPEND)
        write_file(self.syngroups_out, self.syngroups_fl, ACCESS_APPEND)
        write_file(self.index_out, self.index_fl, ACCESS_APPEND)
        print "Done"

    def get_train_tags(self):
        return self.pos_tags

    def set_pos_tags(self,tags):
        self.pos_tags = {}
        word_occurrence = {}
        for word, tag in tags:
            if word in self.pos_tags:
                try:
                    current_tag = self.pos_tags[word]
                    if word_occurrence[word][tag] > word_occurrence[word][current_tag]:
                        print "Word \"" + word +"\" has already been indexed. New POS tag is \"" + tag +"\" replacing \"" + self.pos_tags[word]+"\""
                        self.pos_tags[word] = tag
                    elif tag != current_tag:
                        print "Word \"" + word +"\" has already been indexed. New POS tag is \"" + tag +"\" not replacing current tag \"" + self.pos_tags[word]+"\""
                    word_occurrence[word][tag] += 1
                except KeyError:
                    print "Word \"" + word +"\" has already been indexed. New POS tag is \"" + tag +"\" , not found before. Current tag stays \"" + self.pos_tags[word]+"\""
                    word_occurrence[word][tag] = 1
            else:
                word_occurrence[word] = { tag : 1 }
                self.pos_tags[word] = tag

################################### END GRAMMAR ################################


# Reads a file and stores it in 'content' variable
def read_file(filename):
    f = open(filename, 'r')
    try:
        content = f.read()
    finally:
        f.close()
    return content

# Write to file
def write_file(data,filename,access):
    if access != ACCESS_APPEND and access != ACCESS_OVERWRITE:
        print 'Wrong access type!'
        return
    f = open(filename,access)
    try:
        f.write(data)
    finally:
        f.close()

# Returns a pre-processed list of sentences, separated by new line character
def pre_process(content):
    content = re.sub('[-\+\d\"\.\,\(\)\:\#\$\;\!\']',' ',content)
    return content.lower()

def run_grammar():
    grammar = MyGrammar('../data/vsmall/all.csv')
    grammar.build_grammar()
    return grammar

def run_parser():
    mp = MyParser('../data/small/all.csv','nltk_grammar.cfg')
    mp.main()
    return mp

def run_all():
    grammar = run_grammar()
    parser = run_parser()
    return grammar,parser


def test():
    grammar = nltk.data.load("file:nltk_test.cfg")
    prs = nltk.ChartParser(grammar)
    sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
    trees = prs.nbest_parse(sent)
    return trees
