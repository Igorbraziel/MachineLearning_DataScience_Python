import pickle
from pathlib import Path

algorithm_results_path = Path(__file__).parent.parent.parent / 'algorithm_results.pkl'

with open(algorithm_results_path, 'rb') as file_:
    results_list = pickle.load(file_)


# Algorithms results
decision_tree_results = results_list[0]
random_forest_results = results_list[1]
knn_results = results_list[2]
logistic_regression_results = results_list[3]
svm_results = results_list[4]
neural_network_results = results_list[5]


# ANOVA
from scipy.stats import f_oneway

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.05
_, p = f_oneway(decision_tree_results, random_forest_results, knn_results, logistic_regression_results, svm_results, neural_network_results)

if p <= alpha:
    print('Null hypotesis will be rejected')
    
    algorithms_results = {
        'accuracy': np.concatenate([decision_tree_results, random_forest_results, knn_results, logistic_regression_results, svm_results, neural_network_results]),
        'algorithm': [
            'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree',
            'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree',
            'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree', 'decision_tree',
            'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest',
            'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest',
            'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest', 'random_forest',
            'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn',
            'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn',
            'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn',
            'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression',
            'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression',
            'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression', 'logistic_regression',
            'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm',
            'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm',
            'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm',
            'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network',
            'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network',
            'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network', 'neural_network',
        ]
    }
    
    results_df = pd.DataFrame(algorithms_results)
    
    #Tunkey
    from statsmodels.stats.multicomp import MultiComparison
    
    compare_algorithms = MultiComparison(results_df['accuracy'], results_df['algorithm'])
    
    statistical_test = compare_algorithms.tukeyhsd()
    
    print(statistical_test)
    statistical_test.plot_simultaneous()
    plt.show()
    
else:
    print('Null hypotesis will not be rejected')
