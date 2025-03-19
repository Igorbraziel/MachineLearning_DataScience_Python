import string
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from spacy.lang.pt.stop_words import STOP_WORDS

training_database_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'base_treinamento.txt'

database = pd.read_csv(training_database_path)

sns.countplot(database['emocao'], label='Count', )
plt.show()

print(database.describe())
print()

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

database['texto'] = database['texto'].apply(text_processing)
print(database.head())
print()

final_database = []
for text, emotion in zip(database['texto'], database['emocao']):
    if emotion == 'alegria':
        dictionary = {'ALEGRIA': True, 'MEDO': False}
    else:
        dictionary = {'ALEGRIA': False, 'MEDO': True}
    
    final_database.append([text, dictionary.copy()])
    
print(final_database)    

# Saving the final database in disk
import pickle

final_database_path = Path(__file__).parent.parent.parent / 'final_database.pkl'

with open(final_database_path, 'wb') as f:
    pickle.dump(final_database, f)
    
database_path = Path(__file__).parent.parent.parent / 'database.pkl'

with open(database_path, 'wb') as f:
    pickle.dump(database, f)


    