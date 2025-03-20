import pickle
from pathlib import Path
import random

pickle_database_path = Path(__file__).parent / 'twitter_final_database.pkl'

with open(pickle_database_path, 'rb') as f:
    final_database = pickle.load(f)
    
# Creating the Classifier
import spacy
from spacy.training.example import Example
import numpy as np

model = spacy.blank('pt')
categories = model.add_pipe('textcat')
categories.add_label('POSITIVE')
categories.add_label('NEGATIVE')

BATCH_SIZE = 1000
DROPOUT = 0.2 

historical = []
model.initialize()
for epoch in range(10):
    random.shuffle(final_database)
    losses = {}
    for batch in spacy.util.minibatch(final_database, BATCH_SIZE):
        texts = [text for text, entities in batch]
        annotations = [{'cats': entities} for text, entities in batch]
        
        examples = [
            Example.from_dict(model.make_doc(text), annotation)
            for text, annotation in zip(texts, annotations)
        ]
        
        model.update(examples, losses=losses, drop=DROPOUT)
    print(losses)
    historical.append(losses)
    
        
historical_loss_values = []
for loss in historical:
    historical_loss_values.append(loss.get('textcat'))
    
historical_loss_values = np.array(historical_loss_values)

# Historical losses plot
import matplotlib.pyplot as plt

plt.plot(historical_loss_values)
plt.title('Error progression')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

nlp_model_path = Path(__file__).parent / 'nlp_model'

model.to_disk(nlp_model_path)
