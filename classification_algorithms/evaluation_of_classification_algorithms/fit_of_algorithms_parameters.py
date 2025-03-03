# Fit of algorithms parameters (Grid Search Cross Validation)
from sklearn.model_selection import GridSearchCV

import numpy as np

# Algorithms imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import pickle
from pathlib import Path

credit_data_path = Path(__file__).parent.parent.parent / 'credit.pkl'

with open(credit_data_path, 'rb') as file_:
    X_credit_training, Y_credit_training, X_credit_test, Y_credit_test = pickle.load(file_)
    
X_credit = np.concatenate((X_credit_training, X_credit_test), axis=0)
y_credit = np.concatenate((Y_credit_training, Y_credit_test), axis=0)

if __name__ == '__main__':
    print(X_credit.shape)
    print(y_credit.shape)
    print()
    
# Tuning (afinação) - Decision Tree
parameters = {
    'criterion': ['entropy', 'gini'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

if __name__ == '__main__':
    print('Decision Tree:')
    print(best_parameters)
    print(best_result)
    print('-' * 100)
    
    
# Tuning (afinação) - Random Forest
parameters = {
    'criterion': ['entropy', 'gini'],
    'n_estimators': [10, 40, 100, 150],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

if __name__ == '__main__':
    print('Random Forest:')
    print(best_parameters)
    print(best_result)
    print('-' * 100)  
    
    
# Tuning (afinação) - kNN (K-Nearest-Neighbors)
parameters = {
    'n_neighbors': [2, 5, 10, 30, 50],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 60, 90],
    'p': [1, 2],
    'metric': ['minkowski'],
}

grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

if __name__ == '__main__':
    print('kNN (K-Nearest-Neighbors):')
    print(best_parameters)
    print(best_result)
    print('-' * 100)  
    
    
# Tuning (afinação) - Logistic Regression
parameters = {
    'tol': [0.0001, 0.00001, 0.000001],
    'solver': ['lbfgs', 'sag', 'saga'],
    'C': [1, 1.5, 2],
}

grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

if __name__ == '__main__':
    print('Logistic Regression:')
    print(best_parameters)
    print(best_result)
    print('-' * 100)  
    
    
# Tuning (afinação) - SVC (Support Vector Machines)
parameters = {
    'tol': [0.0001, 0.00001, 0.000001],
    'C': [1, 1.5, 2],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}

grid_search = GridSearchCV(estimator=SVC(), param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

if __name__ == '__main__':
    print('SVC (Support Vector Machines):')
    print(best_parameters)
    print(best_result)
    print('-' * 100)  


# Tuning (afinação) - Neural Networks
parameters = {
    'activation': ['relu', 'logistic', 'tahn'],
    'solver': ['adam', 'sgd'],
    'batch_size': [10, 56]
}

grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

if __name__ == '__main__':
    print('Neural Networks:')
    print(best_parameters)
    print(best_result)
    print('-' * 100)  