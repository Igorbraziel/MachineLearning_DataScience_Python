from pathlib import Path
import pickle

pickle_credit_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(pickle_credit_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_credit_training.shape, Y_credit_training.shape)
    print(X_credit_test.shape, Y_credit_test.shape)
    
# SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

svm_credit = SVC(kernel='rbf', random_state=1, C=50)
svm_credit.fit(X_credit_training, Y_credit_training)

predictions = svm_credit.predict(X_credit_test)

if __name__ == '__main__':
    confusion_matrix = ConfusionMatrix(svm_credit)
    confusion_matrix.fit(X_credit_training, Y_credit_training)
    confusion_matrix.score(X_credit_test, Y_credit_test)
    plt.show()
    
    print(classification_report(Y_credit_test, predictions))
    print(accuracy_score(Y_credit_test, predictions))