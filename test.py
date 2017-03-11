import nltk
import parser

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')

content_text = ' '.join(txt for txt in parser.read_file('../data/tiger/all.csv').split('\r\n'))
tokenized_content = tokenizer.tokenize(content_text)
content_model = nltk.NgramModel(5, tokenized_content)

starting_words = content_model.generate(10000)
content = content_model.generate(5, starting_words)
print ' '.join(content)
