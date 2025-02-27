from pathlib import Path
import pickle

pickle_credit_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(pickle_credit_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_credit_training.shape, Y_credit_training.shape)
    print(X_credit_test.shape, Y_credit_test.shape)
    
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

logistic_regression = LogisticRegression(random_state=1)
logistic_regression.fit(X_credit_training, Y_credit_training)

if __name__ == '__main__':
    print(logistic_regression.intercept_)
    print(logistic_regression.coef_)
    predictions = logistic_regression.predict(X_credit_test)
    print(accuracy_score(Y_credit_test, predictions))
    print(classification_report(Y_credit_test, predictions))
    confusion_matrix = ConfusionMatrix(logistic_regression)
    confusion_matrix.fit(X_credit_training, Y_credit_training)
    confusion_matrix.score(X_credit_test, Y_credit_test)
    plt.show()