import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

pickle_credit_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(pickle_credit_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_credit_training.shape, Y_credit_training.shape)
    print(X_credit_test.shape, Y_credit_test.shape)
    
    
# Random Forest
from sklearn.ensemble import RandomForestClassifier

credit_random_forest = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
credit_random_forest.fit(X_credit_training, Y_credit_training)

predictions = credit_random_forest.predict(X_credit_test)

if __name__ == '__main__':
    print(accuracy_score(Y_credit_test, predictions))
    print(classification_report(Y_credit_test, predictions))
    
    consufion_matrix = ConfusionMatrix(credit_random_forest)
    consufion_matrix.fit(X_credit_training, Y_credit_training)
    consufion_matrix.score(X_credit_test, Y_credit_test)
    plt.show()
