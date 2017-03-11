import parser
from sys import stdout

stdout = open("log.log","w")
grammar = parser.MyGrammar("../data/microwaves/microwaves.csv")
grammar.build_grammar()
