from pathlib import Path
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

pickle_census_path = Path(__file__).parent.parent.parent / 'census.pkl'

with open(pickle_census_path, 'rb') as file_:
    X_census_training, Y_census_training, X_census_test, Y_census_test = pickle.load(file_)
    
naive_census = GaussianNB()
naive_census.fit(X_census_training, Y_census_training)

previsions = naive_census.predict(X_census_test)

if __name__ == '__main__':
    print(accuracy_score(Y_census_test, previsions))
    print(confusion_matrix(Y_census_test, previsions))
    print(classification_report(Y_census_test, previsions))
    
    census_confusion_matrix = ConfusionMatrix(naive_census)
    census_confusion_matrix.fit(X_census_training, Y_census_training)
    census_confusion_matrix.score(X_census_test, Y_census_test)
    plt.show()
    
    
    