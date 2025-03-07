import pickle
from pathlib import Path

import numpy as np

credit_data_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(credit_data_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
X_credit = np.concatenate((X_credit_training, X_credit_test), axis=0)
y_credit = np.concatenate((Y_credit_training, Y_credit_test), axis=0)

print(X_credit.shape, y_credit.shape)

# The best three algorithms
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

neural_network_classifier = MLPClassifier(activation='relu', solver='adam', batch_size=56)
neural_network_classifier.fit(X_credit, y_credit)

random_forest_classifier = RandomForestClassifier(
    n_estimators=150,
    min_samples_leaf=1,
    min_samples_split=5,
    criterion='gini'
)
random_forest_classifier.fit(X_credit, y_credit)

decision_tree_classifier = DecisionTreeClassifier(
    min_samples_leaf=1,
    min_samples_split=5,
    criterion='entropy',
    splitter='best'
)
decision_tree_classifier.fit(X_credit, y_credit)

neural_network_path = Path(__file__).parent.parent.parent / 'neural_network_classifier.sav'
random_forest_path = Path(__file__).parent.parent.parent / 'random_forest_classifier.sav'
decision_tree_path = Path(__file__).parent.parent.parent / 'decision_tree_classifier.sav'

with open(neural_network_path, 'wb') as file_:
    pickle.dump(neural_network_classifier, file_)
    
with open(random_forest_path, 'wb') as file_:
    pickle.dump(random_forest_classifier, file_)
    
with open(decision_tree_path, 'wb') as file_:
    pickle.dump(decision_tree_classifier, file_)
