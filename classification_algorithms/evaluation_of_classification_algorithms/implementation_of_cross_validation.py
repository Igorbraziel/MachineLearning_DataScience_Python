from fit_of_algorithms_parameters import X_credit, y_credit

from sklearn.model_selection import cross_val_score, KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import pandas as pd

decision_tree_results = []
random_forest_results = []
knn_results = []
logistic_regression_results = []
svm_results = []
neural_network_results = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    
    decision_tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=1, 
        min_samples_split=5,
        splitter='best'
    )
    scores = cross_val_score(decision_tree, X_credit, y_credit, cv=kfold)
    decision_tree_results.append(scores.mean())
    
    random_forest = RandomForestClassifier(
        criterion='gini',
        min_samples_leaf=1, 
        min_samples_split=5,
        n_estimators=150,
    )
    scores = cross_val_score(random_forest, X_credit, y_credit, cv=kfold)
    random_forest_results.append(scores.mean())
    
    knn = KNeighborsClassifier(
        algorithm='auto',
        leaf_size=10,
        metric='minkowski',
        n_neighbors=50, 
        p=1,
        weights='distance',
    )
    scores = cross_val_score(knn, X_credit, y_credit, cv=kfold)
    knn_results.append(scores.mean())
    
    logistic_regression = LogisticRegression(
        C=1,
        solver='lbfgs',
        tol=0.0001
    )
    scores = cross_val_score(logistic_regression, X_credit, y_credit, cv=kfold)
    logistic_regression_results.append(scores.mean())
    
    svm = SVC(
        C=1.5,
        kernel='rbf',
        tol=0.0001
    )
    scores = cross_val_score(svm, X_credit, y_credit, cv=kfold)
    svm_results.append(scores.mean())
    
    neural_network = MLPClassifier(
        activation='relu',
        solver='adam',
        batch_size=56
    )
    scores = cross_val_score(neural_network, X_credit, y_credit, cv=kfold)
    neural_network_results.append(scores.mean())
    
    
if __name__ == '__main__':
    results = pd.DataFrame(
        {
            'Decision Tree': decision_tree_results,
            'Random Forest': random_forest_results,
            'KNN': knn_results,
            'Logistic Regression': logistic_regression_results,
            'SVM': svm_results,
            'Neural Network': neural_network_results,
        }
    )
    
    print(results)
    print(results.describe())
    print(f'Variance: {results.var()}')
    print(f'Standard Deviation: {results.std()}')
    print(f'Coefficient of variation: {results.std() / results.mean() * 100}%')
    
    
# Saving the results    
if __name__ == '__main__':
    from pathlib import Path
    import pickle
    
    algorithm_results_path = Path(__file__).parent.parent.parent / 'algorithm_results.pkl'
    
    with open(algorithm_results_path, 'wb') as file_:
        pickle.dump(
            [
                decision_tree_results,
                random_forest_results,
                knn_results,
                logistic_regression_results,
                svm_results,
                neural_network_results,
            ],
            file_
        )