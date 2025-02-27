from pathlib import Path
import pickle

pickle_census_path = Path(__file__).parent.parent.parent / 'census.pkl'

with open(pickle_census_path, 'rb') as file_:
    X_census_training, Y_census_training, X_census_test, Y_census_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_census_training.shape, Y_census_training.shape)
    print(X_census_test.shape, Y_census_test.shape)
    
# Neural Networks
# Multi Layer Perceptron Classifier

from sklearn.neural_network import MLPClassifier

census_neural_network = MLPClassifier(
    max_iter=1000, verbose=True, 
    tol=0.0000100, solver='adam', activation='relu',
    n_iter_no_change=10, hidden_layer_sizes=(55,55)
)
census_neural_network.fit(X_census_training, Y_census_training)

predictions = census_neural_network.predict(X_census_test)

from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

confusion_matrix = ConfusionMatrix(census_neural_network)
confusion_matrix.fit(X_census_training, Y_census_training)
confusion_matrix.score(X_census_test, Y_census_test)
plt.show()

print(classification_report(Y_census_test, predictions))
print(accuracy_score(Y_census_test, predictions))