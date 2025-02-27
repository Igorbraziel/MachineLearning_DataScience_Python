import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

pickle_credit_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(pickle_credit_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
if __name__ == '__main__':
    print(X_credit_training.shape, Y_credit_training.shape)
    print(X_credit_test.shape, Y_credit_test.shape)

credit_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
credit_tree.fit(X_credit_training, Y_credit_training)

forecasters = credit_tree.predict(X_credit_test)

if __name__ == '__main__':
    print(accuracy_score(Y_credit_test, forecasters))
    print(classification_report(Y_credit_test, forecasters))
    confusion_matrix = ConfusionMatrix(credit_tree)
    confusion_matrix.fit(X_credit_training, Y_credit_training)
    confusion_matrix.score(X_credit_test, Y_credit_test)
    plt.show()
    forecasters_names = ['income', 'age', 'loan']
    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
    class_names = [str(x) for x in credit_tree.classes_]
    tree.plot_tree(credit_tree, feature_names=forecasters_names, class_names=class_names, filled=True)
    plt.show()
    figure.savefig('credit_tree.png', format='png')
    
