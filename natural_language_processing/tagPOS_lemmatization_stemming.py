import bs4 as bs
import urllib.request
import nltk
import spacy

nlp = spacy.load('pt_core_news_sm')

document = nlp('Estou aprendendo processamento de linguagem natural, curso em Curitiba')

print(type(document)) 

for token in document:
    print(token.text, token.pos_)
    
print()

for token in document:
    print(token.text, token.lemma_)
    
print()

doc = nlp('encontrei encontraram encontrarão cursei cursava')
print([token.lemma_ for token in doc])
print()

nltk.download('rslp')

stemmer = nltk.stem.RSLPStemmer()
print(stemmer.stem('aprender'))

for token in document:
    print(token.text, stemmer.stem(token.text))