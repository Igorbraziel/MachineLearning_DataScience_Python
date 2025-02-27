from pathlib import Path
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

database_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'risco_credito.csv'

base_credit_risk = pd.read_csv(database_path)

X_credit = base_credit_risk.iloc[:, 0:4].values
Y_credit = base_credit_risk.iloc[:, 4].values

if __name__ == '__main__':
    print(X_credit)

# Converting categorical attributes to numeric
label_encoder_history = LabelEncoder()
label_encoder_debt = LabelEncoder()
label_encoder_guarantees = LabelEncoder()
label_encoder_income = LabelEncoder()

X_credit[:, 0] = label_encoder_history.fit_transform(X_credit[:, 0])
X_credit[:, 1] = label_encoder_debt.fit_transform(X_credit[:, 1])
X_credit[:, 2] = label_encoder_guarantees.fit_transform(X_credit[:, 2])
X_credit[:, 3] = label_encoder_income.fit_transform(X_credit[:, 3])

pickle_credit_risk_path = Path(__file__).parent.parent.parent / 'credit_risk.pkl'

with open(pickle_credit_risk_path, 'wb') as file_:
    pickle.dump([X_credit, Y_credit], file_)

if __name__ == '__main__':
    print(X_credit)
    
    
naive_credit_risk = GaussianNB()
naive_credit_risk.fit(X_credit, Y_credit)

# Samples
# boa (0), alta (0), nenhuma (1), acima_35 (2)
# ruim (2), alta (0), adequada (0), 0_15 (0)

prevision = naive_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

if __name__ == '__main__':
    print(prevision)
    print(naive_credit_risk.classes_)
    print(naive_credit_risk.class_count_)
    print(naive_credit_risk.class_prior_)
    
