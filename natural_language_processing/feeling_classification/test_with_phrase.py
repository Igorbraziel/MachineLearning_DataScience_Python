import spacy
from pathlib import Path
import string
from spacy.lang.pt.stop_words import STOP_WORDS

def text_processing(text: str) -> str:
    text = text.lower()
    nlp = spacy.load('pt_core_news_sm')
    punctuations = string.punctuation
    document = nlp(text)
    
    text_list = []
    for token in document:
        text_list.append(token.lemma_)
        
    text_list: list[str] = [
        word for word in text_list if word not in punctuations and word not in STOP_WORDS and not word.isdigit()
    ]
    
    return ' '.join(text_list)

model_path = Path(__file__).parent.parent.parent / 'felling_classification_model'

model = spacy.load(model_path)

text = input('Enter a phrase to be evaluated: ')

prevision = model(text)
categories = prevision.cats
print('Your phrase feeling is:', 'ALEGRIA' if categories.get('ALEGRIA') > categories.get('MEDO') else 'MEDO')
print(categories)