import pandas as pd
from pathlib import Path
from spacy.lang.pt.stop_words import STOP_WORDS
import string
import spacy
import pickle
import re

# Load database
twitter_database_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'Twitter' / 'Train50.csv'

twitter_database = pd.read_csv(twitter_database_path, encoding='utf-8', delimiter=';')

# Count values (1 - Positive text), (0 - Negative Text)
import seaborn as sns
import matplotlib.pyplot as plt

print(twitter_database.describe())

nlp = spacy.load('pt_core_news_sm')

def text_processing(text: str) -> str:
    text = text.lower()
    text = re.sub(r'@([^\s])+', '', text)
    document = nlp(text)
    text_list = []
    for token in document:
        text_list.append(token.lemma_)
        
    text_list: list[str] = [
        text for text in text_list if text not in string.punctuation and text not in STOP_WORDS and not text.isdigit() 
    ]
    
    return ' '.join(text_list)

twitter_database['tweet_text'] = twitter_database['tweet_text'].apply(text_processing)

print(twitter_database['tweet_text'].head(), twitter_database['tweet_text'].tail(), sep='\n')

with open(Path(__file__).parent / 'twitter_database.pkl', 'wb') as f:
    pickle.dump(twitter_database, f)
    


