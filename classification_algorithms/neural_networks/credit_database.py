from pathlib import Path
import pickle

pickle_credit_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(pickle_credit_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_credit_training.shape, Y_credit_training.shape)
    print(X_credit_test.shape, Y_credit_test.shape)
    
# Neural Networks
# Multi Layer Perceptron Classifier

from sklearn.neural_network import MLPClassifier

credit_neural_network = MLPClassifier(
    max_iter=5000, verbose=True, 
    tol=0.000001, solver='adam', activation='relu',
    random_state=1, n_iter_no_change=1000,
    hidden_layer_sizes=(2,2)
)
credit_neural_network.fit(X_credit_training, Y_credit_training)

predictions = credit_neural_network.predict(X_credit_test)

from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

confusion_matrix = ConfusionMatrix(credit_neural_network)
confusion_matrix.fit(X_credit_training, Y_credit_training)
confusion_matrix.score(X_credit_test, Y_credit_test)
plt.show()

print(classification_report(Y_credit_test, predictions))
print(accuracy_score(Y_credit_test, predictions))