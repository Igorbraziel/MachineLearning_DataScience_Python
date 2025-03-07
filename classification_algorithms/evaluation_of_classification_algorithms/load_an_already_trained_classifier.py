import pickle
from pathlib import Path

import numpy as np

credit_data_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(credit_data_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
X_credit = np.concatenate((X_credit_training, X_credit_test), axis=0)
y_credit = np.concatenate((Y_credit_training, Y_credit_test), axis=0)

print(X_credit.shape, y_credit.shape)

# Loading the classifiers
neural_network_path = Path(__file__).parent.parent.parent / 'neural_network_classifier.sav'
random_forest_path = Path(__file__).parent.parent.parent / 'random_forest_classifier.sav'
decision_tree_path = Path(__file__).parent.parent.parent / 'decision_tree_classifier.sav'

with open(neural_network_path, 'rb') as file_:
    neural_network_classifier = pickle.load(file_)
    
with open(random_forest_path, 'rb') as file_:
    random_forest_classifier = pickle.load(file_)
    
with open(decision_tree_path, 'rb') as file_:
    decision_tree_classifier = pickle.load(file_) 
    
# Testing the classifiers
test_index = 0
new_record: np.ndarray = X_credit[test_index]
new_record = new_record.reshape(1, -1)

print(new_record.shape)

prediction = neural_network_classifier.predict(new_record)
print('Expected:', y_credit[test_index])
print('Received:', prediction)

prediction = random_forest_classifier.predict(new_record)
print('Expected:', y_credit[test_index])
print('Received:', prediction)

prediction = decision_tree_classifier.predict(new_record)
print('Expected:', y_credit[test_index])
print('Received:', prediction)