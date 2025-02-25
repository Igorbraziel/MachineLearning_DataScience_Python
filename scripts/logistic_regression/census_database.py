from pathlib import Path
import pickle

pickle_census_path = Path(__file__).parent.parent.parent / 'census.pkl'

with open(pickle_census_path, 'rb') as file_:
    X_census_training, Y_census_training, X_census_test, Y_census_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_census_training.shape, Y_census_training.shape)
    print(X_census_test.shape, Y_census_test.shape)
    
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

logistic_regression = LogisticRegression(random_state=1)
logistic_regression.fit(X_census_training, Y_census_training)

if __name__ == '__main__':
    print(logistic_regression.intercept_)
    print(logistic_regression.coef_)
    predictions = logistic_regression.predict(X_census_test)
    print(classification_report(Y_census_test, predictions))
    print(accuracy_score(Y_census_test, predictions))
    confusion_matrix = ConfusionMatrix(logistic_regression)
    confusion_matrix.fit(X_census_training, Y_census_training)
    confusion_matrix.score(X_census_test, Y_census_test)
    plt.show()