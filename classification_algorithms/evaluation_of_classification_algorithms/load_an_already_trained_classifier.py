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
test_index = 134
new_record: np.ndarray = X_credit[test_index]
new_record = new_record.reshape(1, -1)

print(new_record.shape)

prediction_neural_network = neural_network_classifier.predict(new_record)[0]
print('Expected:', y_credit[test_index])
print('Received:', prediction_neural_network)

prediction_random_forest = random_forest_classifier.predict(new_record)[0]
print('Expected:', y_credit[test_index])
print('Received:', prediction_random_forest)

prediction_decision_tree = decision_tree_classifier.predict(new_record)[0]
print('Expected:', y_credit[test_index])
print('Received:', prediction_decision_tree)

predictions = [prediction_neural_network, prediction_random_forest, prediction_decision_tree]

# Combination of classifiers
classifiers_combination = {}

for prediction in predictions:
    if not classifiers_combination.get(prediction):
        classifiers_combination[prediction] = 1
    else:
        classifiers_combination[prediction] += 1
        
votes_number = 1
most_votes_prediction = prediction_neural_network

for key, value in classifiers_combination.items():
    if value > votes_number:
        most_votes_prediction = key
        votes_number = value
        
print('Most Votes Prediction:', most_votes_prediction)

# Rejection of classifiers
neural_network_probability = neural_network_classifier.predict_proba(new_record)
neural_network_confidence = neural_network_probability.max()

random_forest_probability = random_forest_classifier.predict_proba(new_record)
random_forest_confidence = random_forest_probability.max()

decision_tree_probability = decision_tree_classifier.predict_proba(new_record)
decision_tree_confidence = decision_tree_probability.max()

used_algorithms = 0
min_confidence = 0.999999
predictions = []

if neural_network_confidence >= min_confidence:
    predictions.append(neural_network_classifier.predict(new_record)[0])
    used_algorithms += 1
    
if random_forest_confidence >= min_confidence:
    predictions.append(random_forest_classifier.predict(new_record)[0])
    used_algorithms += 1
    
if decision_tree_confidence >= min_confidence:
    predictions.append(decision_tree_classifier.predict(new_record)[0])
    used_algorithms += 1


# Combination of classifiers
if used_algorithms > 0:
    classifiers_combination = {}

    for prediction in predictions:
        if not classifiers_combination.get(prediction):
            classifiers_combination[prediction] = 1
        else:
            classifiers_combination[prediction] += 1
            
    votes_number = 1
    most_votes_prediction = predictions[0]

    for key, value in classifiers_combination.items():
        if value > votes_number:
            most_votes_prediction = key
            votes_number = value
            
    print('Rejection of classifiers Prediction:', most_votes_prediction)
    print('Used algorithms:', used_algorithms)
    print(neural_network_confidence, random_forest_confidence, decision_tree_confidence)
else:
    print('No algorithm has been able to classify')