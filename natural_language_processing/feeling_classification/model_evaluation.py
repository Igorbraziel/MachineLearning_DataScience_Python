import pickle
import numpy as np
import spacy
from pathlib import Path
import string
from spacy.lang.pt.stop_words import STOP_WORDS
import pandas as pd

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

database_path = Path(__file__).parent.parent.parent / 'database.pkl'

with open(database_path, 'rb') as f:
    database = pickle.load(f)
    
previsions = []
for text in database['texto']:
    text = text_processing(text)
    prevision = model(text)
    previsions.append(prevision.cats)
    
final_previsions = []
for prevision in previsions:
    if prevision['ALEGRIA'] > prevision['MEDO']:
        final_previsions.append('alegria')
    else:
        final_previsions.append('medo')
        
final_previsions = np.array(final_previsions)

real_results = database['emocao'].values

from sklearn.metrics import confusion_matrix, accuracy_score

print('Accuracy Score (Training):', accuracy_score(real_results, final_previsions))
print('Confusion Matrix (Training):', confusion_matrix(real_results, final_previsions))

# Comparision between results on the test database

test_database = pd.read_csv(Path(__file__).parent.parent.parent / 'Base_de_dados' / 'base_teste.txt', encoding='utf-8')

previsions = []
for text in test_database['emocao']:
    text = text_processing(text)
    prevision = model(text)
    previsions.append(prevision.cats)
    
final_previsions = []
for prevision in previsions:
    if prevision['ALEGRIA'] > prevision['MEDO']:
        final_previsions.append('alegria')
    else:
        final_previsions.append('medo')
        
final_previsions = np.array(final_previsions)

y_test = test_database['emocao'].values

print('Accuracy Score (Test):', accuracy_score(y_test, final_previsions))
print('Confusion Matrix (Test):', confusion_matrix(y_test, final_previsions))


