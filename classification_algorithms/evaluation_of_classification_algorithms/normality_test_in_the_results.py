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

# Normality test
from scipy.stats import shapiro

alpha = 0.05

decision_tree_shapiro = shapiro(decision_tree_results)
random_forest_shapiro = shapiro(random_forest_results)
knn_shapiro = shapiro(knn_results)
logistic_regression_shapiro = shapiro(logistic_regression_results)
svm_shapiro = shapiro(svm_results)
neural_network_shapiro = shapiro(neural_network_results)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.displot(decision_tree_results, kind='kde')
    
    sns.displot(random_forest_results, kind='kde')
    
    sns.displot(knn_results, kind='kde')
    
    sns.displot(logistic_regression_results, kind='kde')
    
    sns.displot(svm_results, kind='kde')
    
    sns.displot(neural_network_results, kind='kde')
    plt.show()
    
    
    print(decision_tree_shapiro)
    print(random_forest_shapiro)
    print(knn_shapiro)
    print(logistic_regression_shapiro)
    print(svm_shapiro)
    print(neural_network_shapiro)