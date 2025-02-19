from division_between_forecaster_and_class import X_credit, Y_credit
from sklearn.preprocessing import StandardScaler

min_income, min_age, min_loan = X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min()

max_income, max_age, max_loan = X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()

if __name__ == '__main__':
    print(min_income, min_age, min_loan)
    print(max_income, max_age, max_loan)
    
# Now, we have to do the standardisation

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

min_income, min_age, min_loan = X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min()

max_income, max_age, max_loan = X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()

if __name__ == '__main__':
    print()
    print(min_income, min_age, min_loan)
    print(max_income, max_age, max_loan)