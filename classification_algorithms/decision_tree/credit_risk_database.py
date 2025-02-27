import pickle
from pathlib import Path

risk_database_path = Path(__file__).parent.parent.parent / 'credit_risk.pkl'

with open(risk_database_path, 'rb') as file_:
    X_credit_risk, Y_credit_risk = pickle.load(file_)
    
    
# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

credit_risk_tree = DecisionTreeClassifier(criterion='entropy')
credit_risk_tree.fit(X_credit_risk, Y_credit_risk)

if __name__ == '__main__':
    print(credit_risk_tree.feature_importances_)
    forecasters = ['historia', 'divida', 'garantias', 'renda']
    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    tree.plot_tree(credit_risk_tree, feature_names=forecasters, class_names=credit_risk_tree.classes_, filled=True)
    plt.show()
    
    # Samples
    # boa (0), alta (0), nenhuma (1), acima_35 (2)
    # ruim (2), alta (0), adequada (0), 0_15 (0)
    forecasts = credit_risk_tree.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
    print(forecasts)
    
