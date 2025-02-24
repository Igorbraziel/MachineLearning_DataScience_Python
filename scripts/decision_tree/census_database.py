import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

pickle_census_path = Path(__file__).parent.parent.parent / 'census.pkl'

with open(pickle_census_path, 'rb') as file_:
    X_census_training, Y_census_training, X_census_test, Y_census_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_census_training.shape, Y_census_training.shape)
    print(X_census_test.shape, Y_census_test.shape)
    
census_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
census_tree.fit(X_census_training, Y_census_training)

forecasters = census_tree.predict(X_census_test)

if __name__ == '__main__':
    print(accuracy_score(Y_census_test, forecasters))
    print(classification_report(Y_census_test, forecasters))
    
    consufion_matrix = ConfusionMatrix(census_tree)
    consufion_matrix.fit(X_census_training, Y_census_training)
    consufion_matrix.score(X_census_test, Y_census_test)
    plt.show() 