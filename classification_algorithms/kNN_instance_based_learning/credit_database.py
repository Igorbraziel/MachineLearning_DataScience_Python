from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import pickle

pickle_credit_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(pickle_credit_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_credit_training.shape, Y_credit_training.shape)
    print(X_credit_test.shape, Y_credit_test.shape)
    
# kNN K-Nearest-Neighbors

knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_credit.fit(X_credit_training, Y_credit_training)

predictions = knn_credit.predict(X_credit_test)

if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, classification_report
    from yellowbrick.classifier import ConfusionMatrix
    import matplotlib.pyplot as plt
    print(classification_report(Y_credit_test, predictions))
    print(accuracy_score(Y_credit_test, predictions))
    confusion_matrix = ConfusionMatrix(knn_credit)
    confusion_matrix.fit(X_credit_training, Y_credit_training)
    confusion_matrix.score(X_credit_test, Y_credit_test)
    plt.show()