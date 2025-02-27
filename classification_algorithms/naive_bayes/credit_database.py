from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from pathlib import Path
import matplotlib.pyplot as plt

root_path = Path(__file__).parent.parent.parent

pickle_credit_path = root_path / 'credit.pkl'

with open(pickle_credit_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
naive_credit = GaussianNB()
naive_credit.fit(X_credit_training, Y_credit_training)

previsions = naive_credit.predict(X_credit_test)

if __name__ == '__main__':
    print(accuracy_score(Y_credit_test, previsions))
    print(confusion_matrix(Y_credit_test, previsions))
    credit_confusion_matrix = ConfusionMatrix(naive_credit)
    credit_confusion_matrix.fit(X_credit_training, Y_credit_training)
    credit_confusion_matrix.score(X_credit_test, Y_credit_test)
    plt.show()
    print(classification_report(Y_credit_test, previsions))