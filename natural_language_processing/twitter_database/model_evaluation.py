from pathlib import Path
import pickle
import spacy
import numpy as np
import pandas as pd

nlp_model_path = Path(__file__).parent / 'nlp_model'

model = spacy.load(nlp_model_path)

twitter_database = pd.read_csv(Path(__file__).parent.parent.parent / 'Base_de_dados' / 'Twitter' / 'Test.csv', encoding='utf-8', delimiter=';')
    
previsions = []
for text in twitter_database['tweet_text']:
    prevision = model(text)
    previsions.append(prevision.cats)
    
final_previsions = []
for cat in previsions:
    value = 1 if cat.get('POSITIVE') > cat.get('NEGATIVE') else 0
    final_previsions.append(value)
  
final_previsions = np.array(final_previsions)
    
y_true = twitter_database['sentiment'].values

from sklearn.metrics import confusion_matrix, accuracy_score
from pprint import pprint

print('Confusion Matrix:', end=' ')
pprint(confusion_matrix(y_true, final_previsions))
print('Accuracy Score:', accuracy_score(y_true, final_previsions))