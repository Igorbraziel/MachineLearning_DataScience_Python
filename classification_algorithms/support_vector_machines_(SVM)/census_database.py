from pathlib import Path
import pickle

pickle_census_path = Path(__file__).parent.parent.parent / 'census.pkl'

with open(pickle_census_path, 'rb') as file_:
    X_census_training, Y_census_training, X_census_test, Y_census_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_census_training.shape, Y_census_training.shape)
    print(X_census_test.shape, Y_census_test.shape)
    
# SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

svm_census = SVC(kernel='linear', random_state=1, C=1.0)
svm_census.fit(X_census_training, Y_census_training)

predictions = svm_census.predict(X_census_test)

if __name__ == '__main__':
    confusion_matrix = ConfusionMatrix(svm_census)
    confusion_matrix.fit(X_census_training, Y_census_training)
    confusion_matrix.score(X_census_test, Y_census_test)
    plt.show()
    
    print(classification_report(Y_census_test, predictions))
    print(accuracy_score(Y_census_test, predictions))