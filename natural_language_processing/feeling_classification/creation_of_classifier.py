import pickle 
from pathlib import Path
import spacy
from spacy.training.example import Example
import numpy as np
import random

final_database_path = Path(__file__).parent.parent.parent / 'final_database.pkl'

with open(final_database_path, 'rb') as f:
    final_database = pickle.load(f)
    
model = spacy.blank('pt')
categories = model.add_pipe('textcat')
categories.add_label('ALEGRIA')
categories.add_label('MEDO')
historical = []

model.begin_training()
for epoch in range(1000):
    random.shuffle(final_database)
    losses = {}
    for batch in spacy.util.minibatch(final_database, 30):
        texts = [
            text for text, entities in batch
        ]
        annotations = [
            {'cats': entities} for text, entities in batch
        ]
        
        examples = [
            Example.from_dict(model.make_doc(text), annotation)
            for text, annotation in zip(texts, annotations)
        ]
        
        model.update(examples, losses=losses)
    if epoch % 100 == 0:
        print(losses)
        historical.append(losses)
        
historical_loss = []
for loss in historical:
    historical_loss.append(loss.get('textcat'))
    
historical_loss = np.array(historical_loss)

import matplotlib.pyplot as plt

plt.plot(historical_loss)
plt.title('Error Progression')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

model.to_disk(Path(__file__).parent.parent.parent / 'felling_classification_model')
    
