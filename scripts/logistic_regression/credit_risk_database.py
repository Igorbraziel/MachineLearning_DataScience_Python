import pickle
from pathlib import Path

risk_database_path = Path(__file__).parent.parent.parent / 'credit_risk.pkl'

with open(risk_database_path, 'rb') as file_:
    X_credit_risk, Y_credit_risk = pickle.load(file_)
    
# Logical Regression
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(random_state=1)
logistic_regression.fit(X_credit_risk, Y_credit_risk)

if __name__ == '__main__':
    print(logistic_regression.intercept_)
    print(logistic_regression.coef_)
    predictions = logistic_regression.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
    print(predictions)